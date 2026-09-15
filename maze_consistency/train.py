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

  a    identity (A):       CE( pi_R(. | h_t, k) , sg[ pi(a | h_t) V_{t+1}(k | h_t, a) / sum_a' ... ] ) for one
                           random step t and bin k per rollout, children from the maze table. Runs as its own
                           a_updates gradient steps per training step, each on a_batch fresh rollouts, with a
                           separate Adam at lr_a; on after a_warmup of training; skipped where the denominator ~0.

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
    a: bool = False
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


def a_batch(tok: Tokenizer, maze, d, idx, rng):
    """Host-side inputs for one A step: NOR prefixes, the same prefixes with MODE = a random bin k, and the four
    children of one random state per rollout (terminal children are labelled analytically)."""
    n, T = len(idx), maze.T
    L = d["length"][idx].astype(np.int64)
    x_nor = tok.with_mode(tok.encode_body(d["positions"][idx], d["actions"][idx], L), None)
    t = (rng.random(n) * L).astype(np.int64)                       # one state per row, t < L
    k = rng.integers(0, maze.K, n)                                  # one outcome bin per row
    x_k = x_nor.copy()
    x_k[:, 0] = tok.mode(k)
    s = d["positions"][idx, t].astype(np.int64)
    si = tok.sidx[t]
    children = np.repeat(x_nor[:, None], N_ACTIONS, 1)              # [n, 4, L, 3]; causal, so later slots don't matter
    nxt = maze.next_open[s]                                         # [n, 4]
    for a in range(N_ACTIONS):
        children[np.arange(n), a, si + 1] = tok.act(np.full(n, a))
        children[np.arange(n), a, si + 2] = tok.pos(nxt[:, a])
    goal = nxt == maze.goal
    return dict(x_nor=x_nor, x_k=x_k, t=t.astype(np.int32), k=k.astype(np.int32),
                children=children.reshape(n * N_ACTIONS, tok.L, 3),
                child_term=goal | ((t + 1) == T)[:, None],
                child_term_bin=np.where(goal, maze.success_bin(t + 1)[:, None], maze.FAIL_BIN).astype(np.int32))


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


def make_a_step(model, tok: Tokenizer, opt_a):
    """One A step: pi_R(. | h_t, k) toward the normalized posterior pi(a | h_t) V_{t+1}(k | child a)."""
    types, sidx = jnp.asarray(tok.types), jnp.asarray(tok.sidx)
    K = tok.K

    def loss_fn(p, ab):
        apply = lambda z: model.apply({"params": p}, z, types)
        n = ab["t"].shape[0]
        st = sidx[ab["t"]]
        pi_t = sg(jax.nn.softmax(apply(ab["x_nor"])["next"][jnp.arange(n), st, :N_ACTIONS], -1))   # [n, 4]
        logpiR = jax.nn.log_softmax(apply(ab["x_k"])["next"][jnp.arange(n), st, :N_ACTIONS], -1)
        vc = apply(ab["children"])["value"][jnp.arange(n * N_ACTIONS), jnp.repeat(st, N_ACTIONS) + 2]
        vc = sg(jax.nn.softmax(vc, -1)).reshape(n, N_ACTIONS, K)
        vc = jnp.where(ab["child_term"][..., None], jax.nn.one_hot(ab["child_term_bin"], K), vc)
        num = pi_t * (vc * jax.nn.one_hot(ab["k"], K)[:, None, :]).sum(-1)
        Z = num.sum(-1)
        ok = Z > 1e-8
        la = -((num / jnp.maximum(Z, 1e-30)[:, None]) * logpiR).sum(-1)
        la = (la * ok).sum() / jnp.maximum(ok.sum(), 1)
        return la, dict(a=la, a_frac=ok.mean())

    @jax.jit
    def a_step(params, opt_state, ab):
        (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, ab)
        u, opt_state = opt_a.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, parts

    return a_step


def train(name="tf", steps=2000, batch=32, lr=1e-3, d_model=64, n_layers=2, n_heads=4, seed=0,
          loss: LossConfig | None = None, consistency=False, a_warmup=None,
          eval_fn=None, eval_every=0, log_every=100, log=print, cons_sampler=None, maze_kw=None,
          mixer=None, mix_frac=0.0):
    """loss: a LossConfig (default: next-token only). consistency=True is shorthand for LossConfig(td=True, a=True).
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
    rollouts here rather than the random-walk training set. Default: uniform from the training set."""
    lc = loss or (LossConfig(td=True, a=True) if consistency else LossConfig())
    if a_warmup is not None:
        lc = replace(lc, a_warmup=a_warmup)
    if mixer is not None and lc.td:
        raise ValueError("td's importance weight assumes the uniform behaviour policy, which model-generated "
                         "rows do not follow -- use mc (or a) with a mixer")
    maze, d = load(**(maze_kw or {}))      # maze_kw={"binning": ..., "n_bins": ...} relabels outcomes only
    tok = Tokenizer(maze)
    n_train = len(d["length"]) - N_HELDOUT
    cfg = ModelConfig.for_tokenizer(tok, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    model = MazeTransformer(cfg)
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(tok.blank(1)), jnp.asarray(tok.types))["params"]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(params)
    step = make_step(model, tok, opt, lc)
    if lc.a:
        opt_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lc.lr_a))
        a_opt_state = opt_a.init(params)
        a_step = make_a_step(model, tok, opt_a)
    fwd = make_forward(model, tok) if eval_fn else None
    log(f"[{name}] {maze} params={count_params(params):,} train={n_train:,} held-out={N_HELDOUT} loss={lc}")
    rng = np.random.default_rng(seed)
    hist, tests, recent, t0 = [], [], {}, time.time()

    def run_eval(i):
        m = dict(step=i, **eval_fn(params, fwd))
        tests.append(m)
        log(f"  test@{i}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "step" and np.isscalar(v)))

    if eval_fn:
        run_eval(0)
    n_nor = batch // 2
    to_jnp = lambda tree: jax.tree_util.tree_map(jnp.asarray, tree)
    for i in range(1, steps + 1):
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
        params, opt_state, parts = step(params, opt_state, jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask),
                                        to_jnp(cb), to_jnp(nb), jnp.float32(cons_on))
        for k, v in parts.items():
            recent.setdefault(k, []).append(float(v))
        if lc.a and i > lc.a_warmup * steps:
            for _ in range(lc.a_updates):
                ab = a_batch(tok, maze, d, rng.integers(0, n_train, lc.a_batch), rng)
                params, a_opt_state, ap = a_step(params, a_opt_state, to_jnp(ab))
                for k, v in ap.items():
                    recent.setdefault(k, []).append(float(v))
        if i % log_every == 0 or i == steps:
            hist.append(dict(step=i, **{k: float(np.mean(v)) for k, v in recent.items()}))
            recent = {}
            log(f"  step {i}: " + " ".join(f"{k} {v:.4f}" for k, v in hist[-1].items() if k != "step")
                + f" ({(time.time() - t0) / i * 1000:.0f} ms/step)")
        if eval_fn and eval_every and (i % eval_every == 0 or i == steps):
            run_eval(i)
    out = os.path.join(RUNS_DIR, name)
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
