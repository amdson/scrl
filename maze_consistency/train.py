"""Training on the canonical rollouts. Every batch is half NOR, half R mode (MODE = the rollout's own outcome
bin) and always trains the teacher-forced next-token loss. LossConfig adds value and consistency terms:

  mc   Monte Carlo value:  CE( V_t , one-hot(outcome bin) ) at every NOR state slot t <= L         (main step)
  td   identity (B):       CE( V_t , sg[ w_t * V_{t+1} ] ), w_t = pi(a_t | h_t) / (1/4) clipped at 4;
                           the last step uses the known outcome                                      (main step)
  cons interval consistency: lambda_cons * L_cons on its own batch of cons_batch rollouts, run in BOTH modes
                           (NOR and the rollout's own bin) so the two orderings share a trajectory. Added into
                           the main gradient, not a separate update. The data loss is untouched, so a cons run
                           and its baseline see an identical data objective and differ only by this term.
                           cons_loss picks the objective from consistency.ALL.       (main step)

  a    legacy oracle-child objective: refused (used real-maze transitions during training).

train() optionally calls eval_fn(params, fwd) every eval_every steps; the results go into history.json.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, asdict, replace

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .dataset import load
from .env import N_ACTIONS
from .tokens import Tokenizer
from .model import ModelConfig, MazeTransformer, make_forward, next_token_loss, count_params
from . import consistency as C

N_HELDOUT = 1000          # the last rollouts of the dataset are held out for evaluation
RUNS_DIR = os.environ.get("RUNS_DIR", "runs")   # set RUNS_DIR (e.g. a Drive folder on Colab) to redirect outputs
sg = jax.lax.stop_gradient


@dataclass(frozen=True)
class LossConfig:
    mc: bool = False
    td: bool = False
    a: bool = False          # legacy flag retained for loading old runs; train() refuses it
    w_mc: float = 1.0
    w_td: float = 1.0
    a_updates: int = 4        # separate A gradient steps per training step
    a_batch: int = 16         # fresh rollouts per A step (one state and one bin each)
    lr_a: float = 1e-4        # A has its own Adam at this rate; the main loss uses train(lr=...)
    a_warmup: float = 0.25    # fraction of training before the A steps start
    cons: bool = False        # interval consistency loss, added into the main gradient
    cons_loss: str = "all_scaled"   # which objective: any key of consistency.ALL
    w_cons: float = 0.1       # lambda_cons
    cons_batch: int = 16      # rollouts per step; each costs two forward passes (NOR and R)
    cons_warmup: float = 0.0  # fraction of training before the consistency term switches on
    cons_detach: str = "none"  # stop-gradient inside the consistency term: none | b | uv | u
                               #   b  : the value head is a fixed teacher; only the token heads move
                               #   uv : the token heads are fixed; only the value head moves
                               # Detaching a whole TERM is well defined under the variance shortcut (it happens
                               # before delta and c are formed). Per-interval teacher/target detachment is not
                               # -- the c_k are shared across every pair (note section 6).

    @property
    def main_value(self) -> bool:
        return self.mc or self.td

    def __post_init__(self):
        if self.cons and self.cons_loss not in C.ALL:
            raise ValueError(f"cons_loss={self.cons_loss!r} not in {sorted(C.ALL)}")
        if self.cons_detach not in ("none", "b", "uv", "u"):
            raise ValueError(f"cons_detach={self.cons_detach!r} not in none|b|uv|u")


def merge_rows(d, idx, extra):
    """Dataset rows idx followed by `extra`, a dict in the same layout (e.g. augment.RolloutBuffer.rows)."""
    return {k: np.concatenate([d[k][idx], np.asarray(extra[k]).astype(d[k].dtype)])
            for k in ("positions", "actions", "length", "reached")}


def make_batch(tok: Tokenizer, maze, d, idx, p_nor=0.5):
    """First round(p_nor * B) rows in NOR mode, the rest with MODE = the rollout's own outcome bin."""
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], d["length"][idx])
    x = tok.with_mode(body, maze.outcome_bin(d["length"][idx], d["reached"][idx]))
    x[:int(round(len(idx) * p_nor)), 0] = tok.mode(None)
    tgt, mask = tok.next_targets(x)
    return x, tgt, mask


def value_batch(maze, d, idx):
    """Host-side inputs for the MC / TD losses on the NOR rows of a batch."""
    L, reached = d["length"][idx].astype(np.int64), d["reached"][idx]
    return dict(act=d["actions"][idx].astype(np.int32), L=L.astype(np.int32),
                term_bin=maze.outcome_bin(L, reached).astype(np.int32))


def cons_batch(tok: Tokenizer, maze, d, idx):
    """Host-side inputs for the consistency term: the same rollouts tokenized in both modes."""
    return C.rollout_batch(tok, maze, d, idx)


def make_step(model, tok: Tokenizer, opt, lc: LossConfig):
    """Main step: next-token loss, the MC / TD value terms on the NOR half, and the consistency term."""
    types, sidx = jnp.asarray(tok.types), jnp.asarray(tok.sidx)
    T, K = tok.T, tok.K
    terms = C.make_terms_fn(model, tok, jit=False) if lc.cons else None

    def cons_term(p, nb):
        """lambda_cons * L_cons on its own batch, both modes on the same rollouts."""
        t = terms(p, nb["x_nor"], nb["x_R"], nb["targets"], nb["R_bin"])
        u, v, b = t["u"], t["v"], t["b"]
        if lc.cons_detach == "b":                 # value head as a fixed teacher
            b = sg(b)
        elif lc.cons_detach == "uv":              # token heads as fixed teachers
            u, v = sg(u), sg(v)
        elif lc.cons_detach == "u":               # only the unconditioned ordering is fixed
            u = sg(u)
        r = C.residuals(u, v, b, nb["lengths"])
        dg = C.diagnostics(r, t["u"], t["v"], t["b"])      # diagnostics always read the undetached terms
        # cond_gap and info_gain are the collapse check: the degenerate optimum of every consistency loss is
        # "ignore R" (v == u, b flat in t), which drives both to 0 while L_cons falls. Logged, never optimized.
        return C.ALL[lc.cons_loss](r).mean(), dict(cond_gap=dg["cond_gap"].mean(),
                                                   info_gain=dg["info_gain"].mean())

    def loss_fn(p, x, tgt, mask, cb, nb, cons_on):
        out = model.apply({"params": p}, x, types)
        tf = next_token_loss(out["next"], tgt, mask)
        total, parts = tf, dict(tf=tf)
        if not lc.main_value:
            return add_cons(p, total, parts, nb, cons_on) if lc.cons else (total, parts)
        n = cb["L"].shape[0]
        logV = jax.nn.log_softmax(out["value"][:n][:, sidx], -1)                   # [n, T+1, K]
        if lc.mc:
            valid_s = jnp.arange(T + 1)[None] <= cb["L"][:, None]
            mc = -(jax.nn.one_hot(cb["term_bin"], K)[:, None] * logV).sum(-1)
            mc = (mc * valid_s).sum() / valid_s.sum()
            total, parts["mc"] = total + lc.w_mc * mc, mc
        if lc.td:
            logpi = jax.nn.log_softmax(out["next"][:n][:, sidx[:T], :N_ACTIONS], -1)
            valid = jnp.arange(T)[None] < cb["L"][:, None]
            lp_a = jnp.take_along_axis(logpi, cb["act"][..., None], -1)[..., 0]
            w = jnp.clip(jnp.exp(sg(lp_a)) * N_ACTIONS, 0.0, 4.0)
            v_next = sg(jnp.exp(logV[:, 1:]))
            last = (jnp.arange(T)[None] + 1) == cb["L"][:, None]
            v_next = jnp.where(last[..., None], jax.nn.one_hot(cb["term_bin"], K)[:, None], v_next)
            td = -(w[..., None] * v_next * logV[:, :T]).sum(-1)
            td = (td * valid).sum() / valid.sum()
            total, parts["td"] = total + lc.w_td * td, td
        return add_cons(p, total, parts, nb, cons_on) if lc.cons else (total, parts)

    def add_cons(p, total, parts, nb, cons_on):
        cons, diag = cons_term(p, nb)
        parts.update(cons=cons, **diag)
        return total + lc.w_cons * cons_on * cons, parts

    @jax.jit
    def step(params, opt_state, x, tgt, mask, cb, nb, cons_on):
        (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, tgt, mask, cb, nb, cons_on)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, parts

    return step


def make_grad_norms(model, tok: Tokenizer, lc: LossConfig):
    """jit'd (params, x, tgt, mask, cb, nb) -> global gradient norm of each loss term taken ALONE (unweighted),
    plus the consistency term split into the recorded-bin rows (the first nb["n_rec"] rows, all rows when the
    key is absent) and the proposal rows, with the per-row consistency loss mean and std of each half. A
    diagnostic for the relative pull of the terms and for the variance of model-proposed queries; it runs
    every train(grad_every=...) steps and never touches the update."""
    types, sidx = jnp.asarray(tok.types), jnp.asarray(tok.sidx)
    T, K = tok.T, tok.K
    terms = C.make_terms_fn(model, tok, jit=False) if lc.cons else None
    gnorm = lambda g: optax.global_norm(g)

    def tf_loss(p, x, tgt, mask):
        return next_token_loss(model.apply({"params": p}, x, types)["next"], tgt, mask)

    def mc_loss(p, x, cb):
        n = cb["L"].shape[0]
        logV = jax.nn.log_softmax(model.apply({"params": p}, x[:n], types)["value"][:, sidx], -1)
        valid_s = jnp.arange(T + 1)[None] <= cb["L"][:, None]
        mc = -(jax.nn.one_hot(cb["term_bin"], K)[:, None] * logV).sum(-1)
        return (mc * valid_s).sum() / valid_s.sum()

    def cons_rows(p, nb):
        t = terms(p, nb["x_nor"], nb["x_R"], nb["targets"], nb["R_bin"])
        return C.ALL[lc.cons_loss](C.residuals(t["u"], t["v"], t["b"], nb["lengths"]))     # per row

    def cons_masked(p, nb, w):
        return (cons_rows(p, nb) * w).sum() / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def f(params, x, tgt, mask, cb, nb):
        out = {"gnorm/tf": gnorm(jax.grad(tf_loss)(params, x, tgt, mask))}
        if lc.mc:
            out["gnorm/mc"] = gnorm(jax.grad(mc_loss)(params, x, cb))
        if lc.cons:
            B = nb["x_nor"].shape[0]
            n_rec = nb["n_rec"] if "n_rec" in nb else B
            rec = (jnp.arange(B) < n_rec).astype(jnp.float32)
            rows = cons_rows(params, nb)
            out["gnorm/cons"] = gnorm(jax.grad(cons_masked)(params, nb, jnp.ones(B, jnp.float32)))
            out["gnorm/cons_recorded"] = gnorm(jax.grad(cons_masked)(params, nb, rec))
            out["gnorm/cons_proposal"] = gnorm(jax.grad(cons_masked)(params, nb, 1.0 - rec))
            for name, w in (("recorded", rec), ("proposal", 1.0 - rec)):
                m = (rows * w).sum() / jnp.maximum(w.sum(), 1.0)
                sd = jnp.sqrt((((rows - m) ** 2) * w).sum() / jnp.maximum(w.sum(), 1.0))
                out[f"cons_rows/{name}_mean"], out[f"cons_rows/{name}_std"] = m, sd
                out[f"cons_rows/{name}_n"] = w.sum()
        return out

    return f


def train(name="tf", steps=2000, batch=32, lr=1e-3, d_model=64, n_layers=2, n_heads=4, seed=0,
          loss: LossConfig | None = None, consistency=False, a_warmup=None,
          eval_fn=None, eval_every=0, log_every=100, log=print, cons_sampler=None, maze_kw=None,
          mixer=None, mix_frac=0.0, init_params=None, metrics_fn=None, grad_every=0, ckpt_every=0,
          resume=True, warmup=0, cosine=False, lr_end_frac=0.1, pos_enc="learned", mode_enc="free"):
    """loss: a LossConfig (default: next-token only). consistency=True is shorthand for LossConfig(mc=True, cons=True).
    eval_fn(params, fwd) -> dict of metrics, called at step 0, every eval_every steps, and at the end.

    mixer / mix_frac: an object with maybe_refresh(params, step), ready and rows(rng, n) -- see
    augment.RolloutBuffer -- supplying model-generated trajectories. Once ready, mix_frac of every main batch
    and every consistency batch comes from it, shuffled across the NOR and R halves, so all heads learn the
    same mixed joint. Without a mixer the data order is exactly what it was.

    maze_kw is forwarded to dataset.load, so a run can use a different outcome binning (n_bins / binning)
    without touching the stored rollouts. Its exact test set has to be rebuilt to match.

    cons_sampler(params, rng, n, step) -> a batch dict like consistency.rollout_batch, overriding where the
    consistency term's rollouts come from. The interval identity constrains the model's own conditionals and
    needs no labels, so it is valid on ANY trajectory distribution -- which is the point of passing model
    rollouts here rather than the random-walk training set. Default: uniform from the training set.

    init_params: start from these parameters (e.g. load_run(name)[0]) instead of a fresh init; the optimizer
    state is fresh either way.

    metrics_fn(step, metrics, kind): optional sink for a logger such as wandb. kind="train" gets the averaged
    loss parts at every log_every steps; kind="test" gets the eval_fn dict at every checkpoint."""
    lc = loss or (LossConfig(mc=True, cons=True) if consistency else LossConfig())
    if a_warmup is not None:
        lc = replace(lc, a_warmup=a_warmup)
    if lc.a:
        raise ValueError("the legacy a objective uses real-maze child transitions and is disabled; "
                         "use mc with interval consistency (cons=True) for offline training")
    if mixer is not None and lc.td:
        raise ValueError("td's importance weight assumes the uniform behaviour policy, which model-generated "
                         "rows do not follow -- use mc with a mixer")
    maze, d = load(**(maze_kw or {}))      # maze_kw={"binning": ..., "n_bins": ...} relabels outcomes only
    tok = Tokenizer(maze)
    n_train = len(d["length"]) - N_HELDOUT
    cfg = ModelConfig.for_tokenizer(tok, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pos_enc=pos_enc,
                                    mode_enc=mode_enc)
    model = MazeTransformer(cfg)
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(tok.blank(1)), jnp.asarray(tok.types))["params"]
    if init_params is not None:
        params = jax.tree_util.tree_map(jnp.asarray, init_params)
    if warmup or cosine:
        warmup = min(int(warmup), max(steps - 1, 0))            # a short run cannot warm up longer than it lasts
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * lr_end_frac) if cosine \
            else optax.linear_schedule(0.0, lr, warmup)
    else:
        sched = lr
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)
    step = make_step(model, tok, opt, lc)
    grad_norms = make_grad_norms(model, tok, lc) if grad_every else None
    fwd = make_forward(model, tok) if eval_fn else None
    log(f"[{name}] maze={maze.H}x{maze.W} T={maze.T} K={maze.K} params={count_params(params):,} "
        f"train={n_train:,} held-out={N_HELDOUT} lr={lr} warmup={warmup} cosine={cosine} pos={pos_enc} mode={mode_enc} loss={lc}")
    rng = np.random.default_rng(seed)
    hist, tests, recent, t0 = [], [], {}, time.time()
    out = os.path.join(RUNS_DIR, name)
    ckpt_path = os.path.join(out, "ckpt.pkl")
    start_step = 1
    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt_state = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, ck["opt_state"])
        hist, tests, start_step = ck["hist"], ck["tests"], ck["step"] + 1
        rng.bit_generator.state = ck["rng_state"]
        log(f"  resumed from {ckpt_path} at step {ck['step']}")

    def save_ckpt(i):
        os.makedirs(out, exist_ok=True)
        tmp = ckpt_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dict(params=jax.device_get(params), opt_state=jax.device_get(opt_state), step=i,
                             hist=hist, tests=tests, rng_state=rng.bit_generator.state), f)
        os.replace(tmp, ckpt_path)

    def run_eval(i):
        m = dict(step=i, **eval_fn(params, fwd))
        tests.append(m)
        if metrics_fn:
            metrics_fn(i, m, "test")
        log(f"  test@{i}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "step" and np.isscalar(v)))

    if eval_fn and start_step == 1:
        run_eval(0)
    n_nor = batch // 2
    to_jnp = lambda tree: jax.tree_util.tree_map(jnp.asarray, tree)
    for i in range(start_step, steps + 1):
        if mixer is not None:
            mixer.maybe_refresh(params, i)
        n_mix = int(round(mix_frac * batch)) if mixer is not None and mixer.ready else 0
        idx = rng.integers(0, n_train, batch - n_mix)
        if n_mix:                                    # rollout rows land in both the NOR and the R half
            src, bidx = merge_rows(d, idx, mixer.rows(rng, n_mix)), rng.permutation(batch)
        else:
            src, bidx = d, idx
        x, tgt, mask = make_batch(tok, maze, src, bidx)
        cb = value_batch(maze, src, bidx[:n_nor]) if lc.main_value else {}
        if not lc.cons:
            nb = {}
        elif cons_sampler is not None:
            nb = cons_sampler(params, rng, lc.cons_batch, i)
        elif n_mix:
            n_c = int(round(mix_frac * lc.cons_batch))
            csrc = merge_rows(d, rng.integers(0, n_train, lc.cons_batch - n_c), mixer.rows(rng, n_c))
            nb = cons_batch(tok, maze, csrc, np.arange(lc.cons_batch))
        else:
            nb = cons_batch(tok, maze, d, rng.integers(0, n_train, lc.cons_batch))
        cons_on = float(lc.cons and i > lc.cons_warmup * steps)
        if grad_norms is not None and i % grad_every == 0:
            for k, v in grad_norms(params, jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask), to_jnp(cb), to_jnp(nb)).items():
                recent.setdefault(k, []).append(float(v))
        params, opt_state, parts = step(params, opt_state, jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask),
                                        to_jnp(cb), to_jnp(nb), jnp.float32(cons_on))
        for k, v in parts.items():
            recent.setdefault(k, []).append(float(v))
        if i % log_every == 0 or i == steps:
            hist.append(dict(step=i, **{k: float(np.mean(v)) for k, v in recent.items()}))
            recent = {}
            if metrics_fn:
                metrics_fn(i, hist[-1], "train")
            log(f"  step {i}: " + " ".join(f"{k} {v:.4f}" for k, v in hist[-1].items() if k != "step")
                + f" ({(time.time() - t0) / (i - start_step + 1) * 1000:.0f} ms/step)")
        if eval_fn and eval_every and (i % eval_every == 0 or i == steps):
            run_eval(i)
        if ckpt_every and i % ckpt_every == 0 and i < steps:
            save_ckpt(i)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump(dict(params=jax.device_get(params), cfg=cfg.__dict__, loss=asdict(lc)), f)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(dict(train=hist, test=tests), f)
    return params, cfg


def load_run(name):
    with open(os.path.join(RUNS_DIR, name, "params.pkl"), "rb") as f:
        z = pickle.load(f)
    return z["params"], ModelConfig(**z["cfg"])
