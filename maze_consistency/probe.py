"""EVALUATION-ONLY probes of a trained model against the real maze. Nothing here may feed training.

Two questions:
  1. Does the model evaluate high-reward trajectories as high reward?  reward_prediction() runs the NOR value
     head along synthetic trajectories with known outcomes (optimal paths, detours, wander-then-optimal, random
     walks) and compares it with the achieved bin and with the exact posterior from the DP.
  2. How much probability does the model put on impossible moves?  impossible_moves() reads the policy (NOR and
     each conditioned bin) for mass on wall-bumps, and the dynamics head for mass off the real next cell.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .env import N_ACTIONS, random_walk_episodes


def _action_to(maze, c, c2):
    """The action that moves c -> c2 (c2 must be a real neighbour)."""
    a = np.flatnonzero(maze.next_open[c] == c2)
    assert len(a), f"{c2} is not a neighbour of {c}"
    return int(a[0])


def _optimal_path(maze, rng, s):
    """Cells from s to the goal along a shortest path, ties broken at random."""
    path = [int(s)]
    while path[-1] != maze.goal:
        c = path[-1]
        nxt = [maze.next_open[c, a] for a in range(N_ACTIONS)]
        best = [n for n in nxt if maze.dist[n] == maze.dist[c] - 1]
        path.append(int(rng.choice(best)))
    return path


def _to_layout(maze, paths):
    """List of cell paths -> dataset layout (positions frozen at the goal, actions after the end 0)."""
    N, T = len(paths), maze.T
    positions = np.full((N, T + 1), maze.goal, np.int32)
    actions = np.zeros((N, T), np.int8)
    length, reached = np.full(N, T, np.int32), np.zeros(N, bool)
    for i, p in enumerate(paths):
        p = p[: T + 1]
        L = len(p) - 1
        positions[i, : L + 1] = p
        for t in range(L):
            actions[i, t] = _action_to(maze, p[t], p[t + 1])
        reached[i] = p[-1] == maze.goal
        length[i] = L if reached[i] else T
        if not reached[i]:
            positions[i, L + 1:] = p[-1]
    return dict(positions=positions, actions=actions, length=length, reached=reached)


def synthetic_trajectories(maze, seed=0, n_per_start=1, detours=(2, 5, 10), wander=(5, 15, 30), n_random=2):
    """Trajectories with known outcomes from EVERY start cell:
        optimal      a shortest path
        detour<k>    a shortest path with k round trips (step off the path and back) inserted at random points
        wander<w>    w uniform-random steps, then a shortest path from wherever that ended
        random       uniform random walks (may time out)
    Returns the dataset layout plus kind [N] (str), start [N], start_dist [N]."""
    rng = np.random.default_rng(seed)
    starts = np.asarray(maze.start_cells)
    paths, kinds = [], []
    for s in starts:
        for _ in range(n_per_start):
            p = _optimal_path(maze, rng, s); paths.append(p); kinds.append("optimal")
            for k in detours:
                q = list(_optimal_path(maze, rng, s))
                for _ in range(k):                           # a round trip at a random point before the goal
                    i = int(rng.integers(0, len(q) - 1))
                    c = q[i]
                    nb = [maze.next_open[c, a] for a in range(N_ACTIONS) if maze.next_open[c, a] != c]
                    q[i:i] = [c, int(rng.choice(nb))]        # ... c, n, c, ... : two extra steps
                if len(q) - 1 <= maze.T:
                    paths.append(q); kinds.append(f"detour{k}")
            for w in wander:
                q = [int(s)]
                for _ in range(w):
                    if q[-1] == maze.goal:
                        break
                    q.append(int(maze.next_open[q[-1], rng.integers(0, N_ACTIONS)]))
                if q[-1] != maze.goal:
                    q += _optimal_path(maze, rng, q[-1])[1:]
                if len(q) - 1 <= maze.T:
                    paths.append(q); kinds.append(f"wander{w}")
    d = _to_layout(maze, paths)
    if n_random:
        rw = random_walk_episodes(maze, n_random * len(starts), seed + 1, starts=np.repeat(starts, n_random))
        for k in ("positions", "actions", "length", "reached"):
            d[k] = np.concatenate([d[k], rw[k].astype(d[k].dtype)])
        kinds += ["random"] * (n_random * len(starts))
    d["kind"] = np.asarray(kinds)
    d["start"] = d["positions"][:, 0].astype(np.int64)
    d["start_dist"] = maze.dist[d["start"]].astype(np.int64)
    d["bin"] = maze.outcome_bin(d["length"], d["reached"]).astype(np.int64)
    return d


def _fwd_chunks(fwd, params, x, key, chunk):
    return np.concatenate([np.asarray(fwd(params, jnp.asarray(x[i:i + chunk]))[key]) for i in range(0, len(x), chunk)]).astype(np.float64)


def _log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def reward_prediction(fwd, params, tok, maze, gt, traj, chunk=64):
    """NOR value head along each trajectory vs the achieved bin and the exact DP posterior.

    Returns per-row arrays over prefixes t = 0..T (nan past the trajectory's end):
        lp_bin [N, T+1]     log q_t(achieved bin)
        p_bin  [N, T+1]     q_t(achieved bin)
        top    [N, T+1]     argmax_k q_t == achieved bin
        kl     [N, T+1]     KL(exact h[t, s_t] || q_t), before arrival only
        remaining [N, T+1]  distance to the goal at s_t
    and scalars per row: terminal_p, terminal_top (the value head at the final prefix)."""
    N, T, K = len(traj["length"]), maze.T, maze.K
    x = tok.encode(traj, R_bin=None)
    logq = _log_softmax(_fwd_chunks(fwd, params, x, "v_logits", chunk))            # [N, T+1, K]
    L = traj["length"].astype(np.int64)
    t = np.arange(T + 1)[None]
    valid = t <= L[:, None]
    pos = traj["positions"].astype(np.int64)
    ach = traj["bin"]
    lp = np.take_along_axis(logq, ach[:, None, None].repeat(T + 1, 1), -1)[..., 0]
    top = logq.argmax(-1) == ach[:, None]
    h = gt.h[np.minimum(t, T), pos]                                                # exact posterior at (t, s_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(h > 0, h * (np.log(h) - logq), 0.0).sum(-1)
    before = valid & (t < L[:, None])                                              # arrival is a determined outcome
    nanify = lambda a, m: np.where(m, a, np.nan)
    rows = np.arange(N)
    return dict(lp_bin=nanify(lp, valid), p_bin=nanify(np.exp(lp), valid), top=nanify(top.astype(float), valid),
                kl=nanify(kl, before), remaining=nanify(maze.dist[pos].astype(float), valid),
                terminal_p=np.exp(lp[rows, L]), terminal_top=top[rows, L].astype(float), q=logq)


def impossible_moves(fwd, params, tok, maze, gt, traj, modes=(None,), chunk=64):
    """Policy mass on wall-bumps and dynamics mass off the real next cell, per mode.

    modes: None for NOR, or a bin index for the conditioned policy. Returns {mode: dict} with per-step arrays
    (nan past the end):
        wall       [N, T]  policy mass on actions that hit a wall at s_t
        wall_exact [N, T]  the same for the exact conditioned random walk pi_R*(bin) (0.25 * n_walls for NOR)
        dyn_wrong  [N, T]  1 - P(real next cell | s_t, a_t) from the dynamics head
    """
    N, T = len(traj["length"]), maze.T
    L = traj["length"].astype(np.int64)
    t = np.arange(T)[None]
    active = t < L[:, None]
    pos, act = traj["positions"].astype(np.int64)[:, :T], traj["actions"].astype(np.int64)
    hits_wall = np.stack([maze.next_open[pos, a] == pos for a in range(N_ACTIONS)], -1)      # [N, T, 4]
    real_next = maze.next_open[pos, act]
    from .dp import truth_for
    gt = truth_for(gt, tok.cond)
    out = {}
    for mode in modes:
        x = tok.encode(traj, R_bin=None if mode is None else np.full(N, mode))
        pi = np.exp(_log_softmax(_fwd_chunks(fwd, params, x, "pi_logits", chunk)))[:, :T]     # [N, T, 4]
        dyn = np.exp(_log_softmax(_fwd_chunks(fwd, params, x, "dyn_logits", chunk)))          # [N, T, cells]
        wall = (pi * hits_wall).sum(-1)
        if mode is None:
            wall_exact = 0.25 * hits_wall.sum(-1)
        else:
            pstar = gt.piR_star[t.repeat(N, 0), pos, mode]                                    # [N, T, 4], nan if h = 0
            wall_exact = (pstar * hits_wall).sum(-1)
        dyn_wrong = 1.0 - np.take_along_axis(dyn, real_next[..., None], -1)[..., 0]
        nanify = lambda a: np.where(active, a, np.nan)
        out[mode] = dict(wall=nanify(wall), wall_exact=nanify(wall_exact), dyn_wrong=nanify(dyn_wrong))
    return out


def by_group(values, groups, order=None):
    """{group: nanmean of values over rows in that group}."""
    order = order or sorted(set(groups.tolist()), key=str)
    return {g: float(np.nanmean(values[groups == g])) for g in order if (groups == g).any()}


def dist_bucket(d, edges=(0, 5, 10, 15, 21)):
    """Start-distance bucket labels such as '5-9'."""
    d = np.asarray(d)
    lab = np.empty(len(d), dtype=object)
    for lo, hi in zip(edges[:-1], edges[1:]):
        lab[(d >= lo) & (d < hi)] = f"{lo}-{hi - 1}"
    return lab.astype(str)


def attainable_bins(maze, start, offsets=(0, 1, 2)):
    """For a start cell, the best bin and the next slower non-empty bins (all attainable within T)."""
    live = [k for k in range(1, maze.K) if k not in maze.empty_bins]
    best = int(maze.best_bin(start))
    below = [k for k in live if k <= best][::-1]                 # best, then slower
    return [below[o] for o in offsets if o < len(below)]


def conditioned_rollouts(params, action_logits, tok, maze, gt, seed=0, n_per_start=4, offsets=(0, 1, 2),
                         greedy=False, include_nor=True):
    """Act in the REAL maze from every start cell, conditioned on an attainable bin, and record what was
    actually achieved. Requests: the start's best bin (offset 0) and the next slower non-empty bins.

    Returns rows (one per rollout) as a dict of arrays: start, start_dist, offset (-1 for NOR), requested,
    achieved, reached, length, norm_return (R / R_opt(start)), got_request (achieved >= requested),
    p_request_rw (exact random-walk probability of achieving >= requested from that start, from the DP)."""
    from .evaluate import rollout
    rng = np.random.default_rng(seed)
    starts = np.repeat(np.asarray(maze.start_cells), n_per_start)
    reqs, offs = [], []
    for s in starts:
        bins = attainable_bins(maze, s, offsets)
        for o, b in zip(offsets, bins):
            reqs.append(b); offs.append(o)
    S = np.repeat(starts, [len(attainable_bins(maze, s, offsets)) for s in starts])
    reqs, offs = np.asarray(reqs), np.asarray(offs)
    if include_nor:
        S = np.concatenate([S, starts]); reqs = np.concatenate([reqs, np.full(len(starts), -1)])
        offs = np.concatenate([offs, np.full(len(starts), -1)])
    ro = rollout(params, action_logits, tok, maze, reqs, S, rng, greedy=greedy)
    ach = maze.outcome_bin(ro["length"], ro["reached"]).astype(np.int64)
    k = np.arange(maze.K)[None]
    h0 = gt.h[0][S]
    p_req = np.where(reqs >= 0, (h0 * (k >= reqs[:, None])).sum(-1), np.nan)
    return dict(start=S, start_dist=maze.dist[S].astype(np.int64), offset=offs, requested=reqs, achieved=ach,
                reached=ro["reached"].astype(bool), length=ro["length"].astype(np.int64),
                norm_return=ro["returns"] / maze.R_opt(S), got_request=(ach >= reqs) & (reqs >= 0),
                p_request_rw=p_req)


# ---- world-model probes (colab/world_model_probe.ipynb) -------------------------------------------------------

def start_values(fwd, params, tok, maze, gt, chunk=64):
    """The NOR value head at prefix 0 (only the start cell is visible) against the exact random walk, for every
    start cell. Expected reward uses the same bin representatives on both sides, so binning cancels:
        model_R  = sum_k q_0(k) r_k              exact_R_binned = sum_k h[0, s, k] r_k
        exact_R  = E[gamma^tau] under the random walk, unbinned (a plain DP; reference only)
        err      = model_R - exact_R_binned      log_ratio = log(model_R / exact_R_binned)
    Returns per-start arrays plus q [S, K] and h0 [S, K]."""
    S = np.asarray(maze.start_cells)
    T = maze.T
    d = dict(positions=np.repeat(S[:, None], T + 1, 1).astype(np.int32), actions=np.zeros((len(S), T), np.int8),
             length=np.zeros(len(S), np.int64))
    x = tok.encode(d, R_bin=None)
    q = np.exp(_log_softmax(_fwd_chunks(fwd, params, x, "v_logits", chunk)[:, 0]))       # [S, K]
    h0 = gt.h[0][S]
    r = maze.bin_reward
    V = np.zeros((T + 1, maze.n_cells)); V[:, maze.goal] = 1.0
    for t in range(T - 1, -1, -1):
        V[t] = np.where(gt.is_goal, 1.0, maze.gamma * V[t + 1][maze.next_open].mean(1))
    model_R, exact_b = q @ r, h0 @ r
    return dict(cell=S, dist=maze.dist[S].astype(np.int64), model_R=model_R, exact_R_binned=exact_b,
                exact_R=V[0][S], err=model_R - exact_b, log_ratio=np.log(model_R) - np.log(exact_b), q=q, h0=h0)


def imagined_transitions(params, state_logits, cell_logits, tok, maze, d, n=2000, seed=0, max_steps=10,
                         modes=(None,), bucket=64, chunk=256):
    """Imagined rollouts (model actions, model dynamics, model END) from random recorded prefixes; every imagined
    transition is checked against the real maze. A prefix is a random dataset row cut at a uniform tau < length.

    Returns one record per imagined step (dict of flat arrays): mode (-1 = NOR), tau, depth (1 = first imagined
    step), cell, action, pred (predicted next cell), real (next_open[cell, action]), ok, and kind:
        0 ok   1 wrong_dir (stayed or moved to an open neighbour that is not where the action leads)
        2 wall (predicted cell is a wall)   3 teleport (an open cell that is not the cell or its neighbour)
    plus per-rollout arrays under "rows": mode, tau, n_steps, any_bad, ended (END emitted).

    Rows are generated `chunk` at a time: each imagined step is one forward pass over the whole chunk, whose
    attention tensor is chunk * heads * L^2 floats with L up to the full 402 slots (uniform tau), so 2000 rows
    at once is ~10 GB whatever the parameter count."""
    from .env import WALL
    from .evaluate import continue_rollout
    rng = np.random.default_rng(seed)
    L = d["length"].astype(np.int64)
    idx = rng.integers(0, len(L), n)
    tau = (rng.random(n) * L[idx]).astype(np.int64)                    # uniform in [0, length)
    pos_src, act_src = d["positions"][idx], d["actions"][idx]
    is_wall = maze.grid.reshape(-1) == WALL
    steps = {k: [] for k in ("mode", "tau", "depth", "cell", "action", "pred", "real", "ok", "kind")}
    rows = {k: [] for k in ("mode", "tau", "n_steps", "any_bad", "ended")}
    t = np.arange(maze.T)[None]
    for mode in modes:
        m = -1 if mode is None else int(mode)
        parts = [continue_rollout(params, state_logits, tok, maze, pos_src[i:i + chunk], act_src[i:i + chunk],
                                  tau[i:i + chunk], np.full(len(tau[i:i + chunk]), m), rng, cell_logits=cell_logits,
                                  learned_end=True, max_steps=max_steps, bucket=bucket) for i in range(0, n, chunk)]
        ro = {k: np.concatenate([p[k] for p in parts]) for k in ("positions", "actions", "length", "reached")}
        pos, act, length = ro["positions"].astype(np.int64), ro["actions"].astype(np.int64), ro["length"].astype(np.int64)
        active = (t >= tau[:, None]) & (t < length[:, None])
        r, c = np.nonzero(active)
        cell, a, pred = pos[r, c], act[r, c], pos[r, c + 1]
        real = maze.next_open[cell, a]
        adjacent = (pred == cell) | (maze.next_open[cell] == pred[:, None]).any(-1)
        kind = np.where(pred == real, 0, np.where(is_wall[pred], 2, np.where(adjacent, 1, 3)))
        for k, v in zip(steps, (np.full(len(r), m), tau[r], c - tau[r] + 1, cell, a, pred, real, kind == 0, kind)):
            steps[k].append(v)
        any_bad = np.zeros(n, bool); np.logical_or.at(any_bad, r, kind > 0)
        for k, v in zip(rows, (np.full(n, m), tau, active.sum(1), any_bad, ro["reached"].astype(bool))):
            rows[k].append(v)
    out = {k: np.concatenate(v) for k, v in steps.items()}
    out["rows"] = {k: np.concatenate(v) for k, v in rows.items()}
    out["kind_names"] = ("ok", "wrong_dir", "wall", "teleport")
    return out
