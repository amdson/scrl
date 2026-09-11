"""Tabular fitted Q-iteration on the same transitions (the cheap reference for 'how bad is
Q-learning here'). Neural FQI lives in train.py (cfg.fqi=True) and shares the transformer trunk."""
from __future__ import annotations

import numpy as np

from .env import Maze, dp_state, rollout_numpy, table_policy, N_ACTIONS
from .dp import GroundTruth
from .eval import summarize_rollout


def transitions_from_data(maze: Maze, data: dict):
    T = maze.T
    n = data["N"]
    s = dp_state(maze, data["positions"], data["flags"])            # [n, T+1]
    valid = np.arange(T)[None] < data["length"][:, None]
    tt = np.broadcast_to(np.arange(T)[None], (n, T))[valid]
    ss = s[:, :T][valid]
    aa = data["actions"][valid].astype(np.int64)
    s2 = s[:, 1:][valid]
    pos2 = data["positions"][:, 1:][valid]
    t2 = tt + 1
    goal = pos2 == maze.goal
    done = goal | (t2 >= T)
    r = -1.0 - ((t2 >= T) & ~goal)
    return dict(t=tt, s=ss, a=aa, t2=t2, s2=s2, r=r, done=done)


def tabular_fqi(maze: Maze, gt: GroundTruth, data: dict, n_iter=200, q_init=0.0, log=None) -> dict:
    T, S = maze.T, gt.S
    tr = transitions_from_data(maze, data)
    counts = np.zeros((T, S, N_ACTIONS))
    np.add.at(counts, (tr["t"], tr["s"], tr["a"]), 1.0)
    visited = counts > 0
    Q = np.full((T + 1, S, N_ACTIONS), q_init)
    Q[T] = 0.0
    hist = []
    for it in range(n_iter):
        y = tr["r"] + np.where(tr["done"], 0.0, Q[tr["t2"], tr["s2"]].max(-1))
        acc = np.zeros((T, S, N_ACTIONS))
        np.add.at(acc, (tr["t"], tr["s"], tr["a"]), y)
        Qn = Q.copy()
        Qn[:T] = np.where(visited, acc / np.maximum(counts, 1), Q[:T])
        delta = np.abs(Qn - Q).max()
        Q = Qn
        m = fqi_metrics(maze, gt, Q, visited)
        m["iter"] = it
        m["delta"] = float(delta)
        hist.append(m)
        if log and it % 50 == 0:
            log(f"  tabular FQI it={it} delta={delta:.4f} q_err={np.nanmean(m['q_err']):.3f} overest={np.nanmean(m['overest']):.3f}")
    greedy = np.eye(N_ACTIONS)[Q[:T].argmax(-1)]
    ro = rollout_numpy(maze, table_policy(greedy, maze), 2000, np.random.default_rng(0))
    return dict(Q=Q, visited=visited, history=hist, final=hist[-1], greedy_rollout=summarize_rollout(maze, ro))


def fqi_metrics(maze: Maze, gt: GroundTruth, Q, visited):
    """Per-t error of max_a Q vs V_opt, weighted by random-walk occupancy d_t over visited states."""
    T = maze.T
    vmax = Q[:T].max(-1)                       # [T, S]
    any_visited = visited.any(-1)
    w = gt.d[:T] * any_visited * (~gt.is_goal)[None]
    err = np.abs(vmax - gt.V_opt[:T])
    over = vmax - gt.V_opt[:T]
    den = w.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_err = np.where(den > 0, (w * err).sum(-1) / den, np.nan)
        overest = np.where(den > 0, (w * over).sum(-1) / den, np.nan)
    return dict(q_err=q_err, overest=overest, coverage=float(any_visited.mean()))


def biased_data(data: dict, drop_frac: float, seed: int = 0) -> dict:
    """Section 9: drop a fraction of the *actions* (transitions are dropped by truncating episodes
    at the first dropped action) to reintroduce coverage gaps. Same data feeds both methods."""
    rng = np.random.default_rng(seed)
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    n, T = d["actions"].shape
    drop = rng.random((n, T)) < drop_frac
    first = np.where(drop.any(1), drop.argmax(1), T)
    newlen = np.minimum(d["length"], first)
    truncated = newlen < d["length"]
    d["length"] = newlen
    d["returns"] = np.where(truncated, -(T + 1), d["returns"])   # truncated episodes count as failures
    return d
