"""Training: configs MC / TD / TD+A, loop closures (E5), and neural FQI. One jit'd step."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .env import Maze, get_maze, random_walk_episodes, N_ACTIONS, FLAG_UNKNOWN, FLAG_OPEN, FLAG_LOCKED
from .dp import compute_ground_truth
from .tokens import Tokenizer, r_distribution
from .model import ModelConfig, MazeTransformer, make_forward, count_params
from .losses import LossConfig, compute_losses, fqi_loss
from .eval import EvalSet, evaluate_model, rollout_model
from .tokens import N_TYPES


@dataclass
class TrainConfig:
    name: str = "run"
    exp: str = "dev"
    loss: str = "TDA"                 # MC | TD | TDA | TDMC
    loss_kw: dict = field(default_factory=dict)
    maze: str = "default"
    N: int = 10000
    data_seed: int = 0
    seed: int = 0
    drop_optimal: bool = True         # remove R = -d* episodes: the optimal return is never observed
    steps: int = 2000
    batch: int = 64
    lr: float = 3e-4
    warmup: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    use_grid: bool = False
    eval_every: int = 250
    eval_n: int = 2000
    rollout_n: int = 512
    rollouts_every_eval: bool = False # rollouts only at the final eval unless True
    log_every: int = 50
    # E5 loop closures
    closure_rollout: bool = False     # 1: prefixes from pi_R(R*) rollouts (off-policy q)
    rollout_every: int = 100
    rollout_buffer: int = 20000
    rollout_frac: float = 0.5
    closure_distill: bool = False     # 2: pi <- pi_R(R*) every distill_every steps
    distill_every: int = 500
    closure_softq: bool = False       # 3: pi_R(R*) inside the (B) backup for bin R*
    # neural FQI baseline
    fqi: bool = False
    target_every: int = 500
    drop_actions: float = 0.0         # section 9: truncate episodes at dropped actions (coverage gaps)
    rtg: bool = False                 # value bins = return-to-go (terminal fact shared across t)
    children: bool = False            # plan-faithful explicit 4-child (B)/(A) on child_batch prefixes per step
    child_batch: int = 32
    out_dir: str = "results"
    save_params: bool = True

    @property
    def run_dir(self):
        return os.path.join(self.out_dir, self.exp, self.name)


class TrainState(train_state.TrainState):
    pass


def make_batch(data, idx, tok: Tokenizer, rng, r_probs, rstar_bin, need_rstar, child_batch=0):
    T = tok.T
    length = data["length"][idx]
    ret_bin = data["returns"][idx] + T + 1
    body = tok.encode_body(data["positions"][idx], data["flags"][idx], data["actions"][idx], length)
    rs_bin = rng.choice(tok.K, size=len(idx), p=r_probs)
    b = dict(
        tok_nor=tok.with_mode(body, None),
        tok_R=tok.with_mode(body, ret_bin),
        tok_Rs=tok.with_mode(body, rs_bin),
        act=data["actions"][idx].astype(np.int32),
        valid_act=(np.arange(T)[None] < length[:, None]),
        valid_state=(np.arange(T + 1)[None] <= length[:, None]),
        term_mask=(np.arange(T + 1)[None] == length[:, None]),
        ret_bin=ret_bin.astype(np.int32),
        rs_bin=rs_bin.astype(np.int32),
        logq=data["logq"][idx].astype(np.float32),
        rstar_bin=np.int32(rstar_bin),
        goal_next=(data["positions"][idx][:, 1:] == tok.maze.goal),
    )
    if need_rstar:
        b["tok_Rstar"] = tok.with_mode(body, np.full(len(idx), rstar_bin))
    if child_batch:
        b["child"] = make_children(tok, data, idx, length, b["tok_nor"], rng, child_batch)
    return b


def make_children(tok: Tokenizer, data, idx, length, tok_nor, rng, n_child):
    """Explicit children for a random subset of (episode, t<L) prefixes in the batch.
    Returns idx [Bc], t [Bc], tok [Bc*A*F, L] (NOR mode, prefix truncated at t, then (a, pos')),
    w [Bc,A,F] branch weights, goal/timeout [Bc,A,F] bool (analytic terminal)."""
    maze = tok.maze
    T, A = tok.T, N_ACTIONS
    F = 2 if maze.has_door else 1
    B = len(idx)
    ci = rng.integers(0, B, min(n_child, B))
    ct = (rng.random(len(ci)) * length[ci]).astype(np.int64)            # t < L
    Bc = len(ci)
    pos = data["positions"][idx][ci, ct]
    flag = data["flags"][idx][ci, ct]
    toks = np.repeat(tok_nor[ci][:, None, None, :], A, 1).repeat(F, 2).copy()   # [Bc,A,F,L]
    # truncate after the state token at t
    cut = tok.sidx[ct] + 1                                               # [Bc]
    ar = np.arange(tok.L)[None, None, None, :]
    toks = np.where(ar >= cut[:, None, None, None], tok.PAD, toks)
    w = np.zeros((Bc, A, F))
    goal = np.zeros((Bc, A, F), bool)
    for a in range(A):
        for f in range(F):
            nxt_tab = maze.next_open if f == 0 else maze.next_locked
            npos = nxt_tab[pos, a]
            nflag = flag.copy()
            wf = np.ones(Bc)
            if maze.has_door:
                attempted = (maze.next_open[pos, a] == maze.door) & (pos != maze.door)
                unk = flag == FLAG_UNKNOWN
                if f == 0:
                    nflag = np.where(npos == maze.door, FLAG_OPEN, flag)
                    wf = np.where(flag == FLAG_LOCKED, 0.0, np.where(unk & attempted, 1 - maze.p_locked, 1.0))
                else:
                    nflag = np.where(attempted, FLAG_LOCKED, flag)
                    wf = np.where(flag == FLAG_LOCKED, 1.0, np.where(unk & attempted, maze.p_locked, 0.0))
            toks[np.arange(Bc), a, f, tok.sidx[ct] + 1] = tok.ACT0 + a
            toks[np.arange(Bc), a, f, tok.sidx[ct] + 2] = tok.pos_token(npos, nflag)
            w[:, a, f] = wf
            goal[:, a, f] = npos == maze.goal
    timeout = (~goal) & ((ct + 1) >= T)[:, None, None]
    return dict(idx=ci.astype(np.int32), t=ct.astype(np.int32), tok=toks.reshape(Bc * A * F, tok.L),
                w=w.astype(np.float32), goal=goal, timeout=timeout)


def make_train_step(model, tok: Tokenizer, lcfg: LossConfig, need_rstar: bool, need_teacher: bool):
    types = jnp.asarray(tok.types)
    sidx = jnp.asarray(tok.sidx)
    names = ["nor", "R", "Rs"] + (["Rstar"] if need_rstar else [])

    def loss_fn(params, batch, step, teacher_params):
        seqs = [batch["tok_nor"], batch["tok_R"], batch["tok_Rs"]] + ([batch["tok_Rstar"]] if need_rstar else [])
        toks = jnp.concatenate(seqs, 0)
        out = model.apply({"params": params}, toks, types)
        g = {k: v[:, sidx] for k, v in out.items()}
        B = batch["act"].shape[0]
        heads = {n: {k: g[k][i * B:(i + 1) * B] for k in g} for i, n in enumerate(names)}
        if "child" in batch:
            cout = model.apply({"params": params}, batch["child"]["tok"], types)
            heads["child"] = {k: jax.lax.stop_gradient(v[:, sidx]) for k, v in cout.items()}
        teacher_heads = None
        if need_teacher:
            tout = model.apply({"params": teacher_params}, batch["tok_Rstar"], types)
            teacher_heads = {"Rstar": {k: jax.lax.stop_gradient(v[:, sidx]) for k, v in tout.items()}}
        return compute_losses(heads, batch, lcfg, step, teacher_heads)

    @jax.jit
    def step_fn(state, batch, teacher_params):
        (total, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, state.step, teacher_params)
        losses["grad_norm"] = optax.global_norm(grads)
        return state.apply_gradients(grads=grads), losses

    return step_fn


def make_fqi_step(model, tok: Tokenizer, maze_T: int):
    types = jnp.asarray(tok.types)
    sidx = jnp.asarray(tok.sidx)

    def loss_fn(params, batch, target_params):
        out = model.apply({"params": params}, batch["tok_nor"], types)
        tout = model.apply({"params": target_params}, batch["tok_nor"], types)
        q = {k: v[:, sidx] for k, v in out.items()}
        qt = {k: jax.lax.stop_gradient(v[:, sidx]) for k, v in tout.items()}
        return fqi_loss(q, qt, batch, maze_T, batch["goal_next"])

    @jax.jit
    def step_fn(state, batch, target_params):
        (total, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, target_params)
        losses["grad_norm"] = optax.global_norm(grads)
        losses["total"] = total
        return state.apply_gradients(grads=grads), losses

    return step_fn


def build(cfg: TrainConfig, maze: Maze | None = None):
    maze = maze or get_maze(cfg.maze)
    gt = compute_ground_truth(maze)
    tok = Tokenizer(maze, cfg.use_grid)
    mcfg = ModelConfig(vocab=tok.vocab, n_types=N_TYPES, max_len=tok.L, K=maze.K,
                       d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    model = MazeTransformer(mcfg)
    key = jax.random.PRNGKey(cfg.seed)
    params = model.init(key, jnp.zeros((1, tok.L), jnp.int32), jnp.asarray(tok.types))["params"]
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, cfg.warmup, max(cfg.steps, cfg.warmup + 1), cfg.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(cfg.clip), optax.adamw(sched, weight_decay=cfg.weight_decay))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    fwd = make_forward(model, tok.types, tok.sidx)
    return maze, gt, tok, model, state, fwd


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in x.tolist()]
    if isinstance(x, (np.floating, jnp.ndarray)):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def train(cfg: TrainConfig, log=print, maze: Maze | None = None) -> dict:
    t0 = time.time()
    maze, gt, tok, model, state, fwd = build(cfg, maze)
    lcfg = LossConfig.named(cfg.loss, **cfg.loss_kw) if not cfg.fqi else LossConfig()
    lcfg.rtg = cfg.rtg
    lcfg.use_children = cfg.children
    child_batch = cfg.child_batch if cfg.children else 0
    if cfg.closure_softq:
        lcfg.td_backup = "piR"
    if cfg.closure_distill:
        lcfg.distill = True
    need_rstar = cfg.closure_softq or cfg.closure_distill
    need_teacher = cfg.closure_distill
    rstar_bin = int(maze.bin_of(maze.R_max))
    data = random_walk_episodes(maze, cfg.N, cfg.data_seed, cfg.drop_optimal)
    if cfg.drop_actions > 0:
        from .fqi import biased_data
        data = biased_data(data, cfg.drop_actions, cfg.data_seed)
    r_probs = r_distribution(maze)
    es = EvalSet(maze, gt, tok, n=cfg.eval_n, seed=10_000 + cfg.data_seed)
    rng = np.random.default_rng(cfg.seed + 777)
    step_fn = make_fqi_step(model, tok, maze.T) if cfg.fqi else make_train_step(model, tok, lcfg, need_rstar, need_teacher)
    log(f"[{cfg.exp}/{cfg.name}] {maze} params={count_params(state.params):,} data N={data['N']} "
        f"(solve {np.mean(data['returns'] > -(maze.T+1)):.3f}, optimal {np.mean(data['returns'] == maze.R_max):.4f}) L={tok.L}")
    os.makedirs(cfg.run_dir, exist_ok=True)
    history, evals = [], []
    teacher = state.params          # closure 2 teacher / FQI target network
    buffer = None                   # closure 1 rollout buffer
    losses_acc = {}

    def do_eval(step, final=False):
        m = evaluate_model(state.params, fwd, es, rollouts=(final or cfg.rollouts_every_eval),
                           rollout_n=cfg.rollout_n, seed=cfg.seed, fqi=cfg.fqi, detail=final, rtg=cfg.rtg)
        m["step"] = step
        m["wall"] = time.time() - t0
        evals.append(m)
        s = (f"  eval@{step}: value_err mean={m['mean_value_err']:.4f} max={m['max_value_err']:.4f} "
             f"opt_err={m['opt_err']:.3f} cons={m['mean_consistency']:.3f} piR_err={m['mean_piR_err']:.3f}")
        if cfg.fqi:
            s += f" q_err={np.nanmean(m['q_err']):.3f} overest={np.nanmean(m['overest']):.3f}"
        if "rollouts" in m:
            s += " | " + " ".join(f"{k}:{v['solve_rate']:.2f}/{v['optimal_rate']:.2f}" for k, v in m["rollouts"].items())
        log(s)
        return m

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step, final=(step == cfg.steps))
            if step == cfg.steps:
                break
        # closure 1: refresh the rollout buffer from pi_R(R*)
        if cfg.closure_rollout and step % cfg.rollout_every == 0:
            ro = rollout_model(state.params, fwd, tok, maze, policy="piR", N=min(1024, cfg.rollout_buffer), rng=rng)
            # behaviour log-prob of the taken actions under pi_R(R*): recompute from the model
            out = fwd(state.params, jnp.asarray(tok.with_mode(tok.encode_body(ro["positions"], ro["flags"], ro["actions"], ro["length"]), rstar_bin)))
            lp = jax.nn.log_softmax(out["action"][:, :maze.T], -1)
            ro["logq"] = np.asarray(jnp.take_along_axis(lp, jnp.asarray(ro["actions"].astype(np.int32))[..., None], -1)[..., 0])
            ro["N"] = ro["returns"].shape[0]
            if buffer is None:
                buffer = ro
            else:
                buffer = {k: (np.concatenate([buffer[k], ro[k]])[-cfg.rollout_buffer:] if isinstance(ro[k], np.ndarray) else ro[k]) for k in ro}
                buffer["N"] = buffer["returns"].shape[0]
        # closure 2 / FQI: refresh teacher / target network
        if (cfg.closure_distill and step % cfg.distill_every == 0) or (cfg.fqi and step % cfg.target_every == 0):
            teacher = jax.tree_util.tree_map(lambda x: x, state.params)
        # batch
        if buffer is not None and cfg.closure_rollout:
            nb = int(cfg.batch * cfg.rollout_frac)
            i1 = rng.integers(0, data["N"], cfg.batch - nb)
            i2 = rng.integers(0, buffer["N"], nb)
            b1 = make_batch(data, i1, tok, rng, r_probs, rstar_bin, need_rstar)
            b2 = make_batch(buffer, i2, tok, rng, r_probs, rstar_bin, need_rstar)
            batch = {k: (np.concatenate([b1[k], b2[k]]) if isinstance(b1[k], np.ndarray) and b1[k].ndim > 0 else b1[k]) for k in b1}
            if child_batch:
                mixed = {k: np.concatenate([data[k][i1], buffer[k][i2]]) for k in ("positions", "flags", "actions", "length")}
                batch["child"] = make_children(tok, mixed, np.arange(cfg.batch), batch["valid_act"].sum(1), batch["tok_nor"], rng, child_batch)
        else:
            idx = rng.integers(0, data["N"], cfg.batch)
            batch = make_batch(data, idx, tok, rng, r_probs, rstar_bin, need_rstar, child_batch)
        state, losses = step_fn(state, batch, teacher)
        for k, v in losses.items():
            losses_acc.setdefault(k, []).append(float(v))
        if step % cfg.log_every == 0 and step > 0:
            rec = {k: float(np.mean(v)) for k, v in losses_acc.items()}
            rec["step"] = step
            history.append(rec)
            losses_acc = {}
            if step % (cfg.log_every * 10) == 0:
                log(f"  step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k != "step")
                    + f" ({(time.time()-t0)/max(step,1)*1000:.0f} ms/step)")

    result = dict(config=asdict(cfg), maze=dict(name=maze.name, d_star=maze.d_star, T=maze.T, K=maze.K,
                  ascii=maze.ascii(), n_cells=maze.n_cells, H=maze.H, W=maze.W),
                  data=dict(N=data["N"], solve=float(np.mean(data["returns"] > -(maze.T + 1))),
                            optimal=float(np.mean(data["returns"] == maze.R_max))),
                  history=history, evals=evals, wall=time.time() - t0)
    with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as f:
        json.dump(_to_jsonable(result), f)
    if evals and "_detail" in evals[-1]:
        np.savez(os.path.join(cfg.run_dir, "detail.npz"), **{k: np.asarray(v, dtype=object) if isinstance(v, list) else v
                                                             for k, v in evals[-1]["_detail"].items()})
    if cfg.save_params:
        with open(os.path.join(cfg.run_dir, "params.pkl"), "wb") as f:
            pickle.dump(jax.device_get(state.params), f)
    log(f"[{cfg.exp}/{cfg.name}] done in {time.time()-t0:.0f}s")
    return result


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        res = json.load(f)
    params = None
    p = os.path.join(run_dir, "params.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            params = pickle.load(f)
    return res, params
