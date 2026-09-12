"""Training on the canonical rollouts. Every batch is half NOR, half R mode (MODE = the rollout's own outcome
bin) and always trains the teacher-forced next-token loss. LossConfig adds value and consistency terms, all on
the NOR half of the batch:

  mc   Monte Carlo value:  CE( V_t , one-hot(outcome bin) ) at every state slot t <= L
  td   identity (B):       CE( V_t , sg[ w_t * V_{t+1} ] ), w_t = pi(a_t | h_t) / (1/4) clipped at 4;
                           the last step uses the known outcome
  a    identity (A):       one random step t and bin k per sequence, explicit children from the maze table:
                           CE( pi_R(. | h_t, k) , sg[ pi(a | h_t) V_{t+1}(k | h_t, a) / sum_a' ... ] ),
                           switched on after a_warmup of training; skipped where the denominator is ~0

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
    w_a: float = 1.0
    a_warmup: float = 0.25    # fraction of training before the A loss switches on

    @property
    def needs_value(self) -> bool:
        return self.mc or self.td or self.a


def make_batch(tok: Tokenizer, maze, d, idx, p_nor=0.5):
    """First round(p_nor * B) rows in NOR mode, the rest with MODE = the rollout's own outcome bin."""
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], d["length"][idx])
    x = tok.with_mode(body, maze.outcome_bin(d["length"][idx], d["reached"][idx]))
    x[:int(round(len(idx) * p_nor)), 0] = tok.mode(None)
    tgt, mask = tok.next_targets(x)
    return x, tgt, mask


def consistency_batch(tok: Tokenizer, maze, d, idx, x_nor, rng, lc: LossConfig):
    """Host-side inputs for the value and A losses on NOR rows x_nor [n, L, 3]."""
    L, reached = d["length"][idx].astype(np.int64), d["reached"][idx]
    cb = dict(act=d["actions"][idx].astype(np.int32), L=L.astype(np.int32),
              term_bin=maze.outcome_bin(L, reached).astype(np.int32))
    if not lc.a:
        return cb
    n, T = len(idx), maze.T
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
    cb.update(t=t.astype(np.int32), k=k.astype(np.int32), x_k=x_k,
              children=children.reshape(n * N_ACTIONS, tok.L, 3),
              child_term=goal | ((t + 1) == T)[:, None],
              child_term_bin=np.where(goal, maze.success_bin(t + 1)[:, None], maze.FAIL_BIN).astype(np.int32))
    return cb


def make_step(model, tok: Tokenizer, opt, lc: LossConfig):
    types, sidx = jnp.asarray(tok.types), jnp.asarray(tok.sidx)
    T, K = tok.T, tok.K

    def loss_fn(p, x, tgt, mask, cb, a_on):
        apply = lambda z: model.apply({"params": p}, z, types)
        out = apply(x)
        tf = next_token_loss(out["next"], tgt, mask)
        total, parts = tf, dict(tf=tf)
        if not lc.needs_value:
            return total, parts
        n = cb["L"].shape[0]
        logpi = jax.nn.log_softmax(out["next"][:n][:, sidx, :N_ACTIONS], -1)      # [n, T+1, 4]
        logV = jax.nn.log_softmax(out["value"][:n][:, sidx], -1)                   # [n, T+1, K]
        if lc.mc:
            valid_s = jnp.arange(T + 1)[None] <= cb["L"][:, None]
            mc = -(jax.nn.one_hot(cb["term_bin"], K)[:, None] * logV).sum(-1)
            mc = (mc * valid_s).sum() / valid_s.sum()
            total, parts["mc"] = total + lc.w_mc * mc, mc
        if lc.td:
            valid = jnp.arange(T)[None] < cb["L"][:, None]
            lp_a = jnp.take_along_axis(logpi[:, :T], cb["act"][..., None], -1)[..., 0]
            w = jnp.clip(jnp.exp(sg(lp_a)) * N_ACTIONS, 0.0, 4.0)
            v_next = sg(jnp.exp(logV[:, 1:]))
            last = (jnp.arange(T)[None] + 1) == cb["L"][:, None]
            v_next = jnp.where(last[..., None], jax.nn.one_hot(cb["term_bin"], K)[:, None], v_next)
            td = -(w[..., None] * v_next * logV[:, :T]).sum(-1)
            td = (td * valid).sum() / valid.sum()
            total, parts["td"] = total + lc.w_td * td, td
        if lc.a:
            st = sidx[cb["t"]]
            logpiR = jax.nn.log_softmax(apply(cb["x_k"])["next"][jnp.arange(n), st, :N_ACTIONS], -1)   # [n, 4]
            vc = apply(cb["children"])["value"][jnp.arange(n * N_ACTIONS), jnp.repeat(st, N_ACTIONS) + 2]
            vc = sg(jax.nn.softmax(vc, -1)).reshape(n, N_ACTIONS, K)
            vc = jnp.where(cb["child_term"][..., None], jax.nn.one_hot(cb["child_term_bin"], K), vc)
            pi_t = sg(jnp.exp(logpi[jnp.arange(n), cb["t"]]))                      # [n, 4]
            num = pi_t * (vc * jax.nn.one_hot(cb["k"], K)[:, None, :]).sum(-1)
            Z = num.sum(-1)
            ok = Z > 1e-8
            la = -((num / jnp.maximum(Z, 1e-30)[:, None]) * logpiR).sum(-1)
            la = (la * ok).sum() / jnp.maximum(ok.sum(), 1)
            total = total + a_on * lc.w_a * la
            parts.update(a=la, a_frac=ok.mean())
        return total, parts

    @jax.jit
    def step(params, opt_state, x, tgt, mask, cb, a_on):
        (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, tgt, mask, cb, a_on)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, parts

    return step


def train(name="tf", steps=2000, batch=32, lr=1e-3, d_model=64, n_layers=2, n_heads=4, seed=0,
          loss: LossConfig | None = None, consistency=False, a_warmup=None,
          eval_fn=None, eval_every=0, log_every=100, log=print):
    """loss: a LossConfig (default: next-token only). consistency=True is shorthand for LossConfig(td=True, a=True).
    eval_fn(params, fwd) -> dict of metrics, called at step 0, every eval_every steps, and at the end."""
    lc = loss or (LossConfig(td=True, a=True) if consistency else LossConfig())
    if a_warmup is not None:
        lc = replace(lc, a_warmup=a_warmup)
    maze, d = load()
    tok = Tokenizer(maze)
    n_train = len(d["length"]) - N_HELDOUT
    cfg = ModelConfig.for_tokenizer(tok, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    model = MazeTransformer(cfg)
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(tok.blank(1)), jnp.asarray(tok.types))["params"]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(params)
    step = make_step(model, tok, opt, lc)
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
    for i in range(1, steps + 1):
        idx = rng.integers(0, n_train, batch)
        x, tgt, mask = make_batch(tok, maze, d, idx)
        cb = consistency_batch(tok, maze, d, idx[:n_nor], x[:n_nor], rng, lc) if lc.needs_value else {}
        a_on = jnp.float32(i > lc.a_warmup * steps)
        params, opt_state, parts = step(params, opt_state, jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask),
                                        jax.tree_util.tree_map(jnp.asarray, cb), a_on)
        for k, v in parts.items():
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
