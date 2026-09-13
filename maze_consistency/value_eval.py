"""Exact expected return of any rollout prefix under the random walk, and an eval of the value head against it.

A prefix at cell s at step t that has not reached the goal continues as a uniform random walk from s with T - t
steps left, so (R = gamma**L from the rollout's own start, 0 if the goal is not reached by T)
    E[R | prefix] = gamma**t * u_{T-t}(s),      u_n(s) = E[gamma**tau(s) ; tau(s) <= n]
with tau(s) the random walk's hitting time of the goal from s. u_n is the n-step truncation of the linear system
    u(goal) = 1,   u(s) = gamma * mean_a u(next(s, a)),     i.e.  (I - gamma P_QQ) u_Q = gamma P_Qg
over the open non-goal cells Q. hitting_value solves that system directly (the T - t -> inf limit);
expected_return_table iterates it for n = 0..T, which is exact under the horizon. A prefix that reached the goal
at step L has E[R | prefix] = gamma**L.

value_accuracy scores the model's value head (MODE = NOR) at every prefix of a set of trajectories against this.
The model's E[R] is sum_k V_t(k | prefix) * bin_reward[k]. A K-bin head cannot represent E[R] exactly even with the
exact bin distribution h (dp.py), so h's error under the same readout is reported next to the model's as "bins".
"""
from __future__ import annotations

import numpy as np

from .env import Maze, N_ACTIONS, WALL
from .dp import compute_ground_truth
from .tokens import Tokenizer


def hitting_value(maze: Maze) -> np.ndarray:
    """E[gamma**tau(s)] of the untruncated random walk, [S], from one linear solve (0 on walls)."""
    S, g = maze.n_cells, maze.goal
    Q = np.flatnonzero((maze.grid.reshape(-1) != WALL) & (np.arange(S) != g))
    P = np.zeros((S, S))
    np.add.at(P, (np.repeat(np.arange(S), N_ACTIONS), maze.next_open.reshape(-1)), 1.0 / N_ACTIONS)
    u = np.zeros(S)
    u[g] = 1.0
    u[Q] = np.linalg.solve(np.eye(len(Q)) - maze.gamma * P[np.ix_(Q, Q)], maze.gamma * P[Q, g])
    return u


def expected_return_table(maze: Maze) -> np.ndarray:
    """V[t, s] = E[R | at s at step t, goal not reached before], [T+1, S]. V[t, goal] = gamma**t (arrived at t);
    V[T, s] = 0 elsewhere (timed out)."""
    T, g = maze.T, maze.goal
    u = np.zeros(maze.n_cells)                     # u_0: only the goal pays
    u[g] = 1.0
    V = np.zeros((T + 1, maze.n_cells))
    V[T] = u
    for t in range(T - 1, -1, -1):                 # V[t] holds u_{T-t}
        u = maze.gamma * u[maze.next_open].mean(1)
        u[g] = 1.0
        V[t] = u
    return V * maze.gamma ** np.arange(T + 1)[:, None]


def prefix_expected_return(maze: Maze, positions, length, V=None) -> np.ndarray:
    """E[R | prefix up to step t] for every prefix t = 0..T of each trajectory, [N, T+1]; nan for t > length."""
    V = expected_return_table(maze) if V is None else V
    t = np.arange(maze.T + 1)
    v = V[t[None], np.asarray(positions, dtype=np.int64)]
    return np.where(t[None] <= np.asarray(length)[:, None], v, np.nan)


def _log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def value_accuracy(params, fwd, tok: Tokenizer, trajs=None, rows=None, chunk=200, detail=False) -> dict:
    """Value head vs the exact E[R] at every prefix t <= L of `trajs`, read with MODE = NOR. trajs: a dict with
    positions [N, T+1], actions [N, T], length [N] and optionally setting / setting_names (default: the exact test
    set). Metrics per setting and over "all" prefixes:
        ev_mae       mean |E_model[R] - E[R]|
        ev_logerr    mean |log E_model[R] - log E[R]| in nats, over prefixes with E[R] > 0
        ev_bias      mean (log E_model[R] - log E[R]); > 0 means the model overestimates
        bins_mae, bins_logerr   the same for the exact bin distribution h: the floor for a K-bin value head
    Returns flat "<metric>/<setting>" keys and "per_setting" (like testset.score); with detail=True also "prefix",
    per-prefix arrays [n, T+1] (true, model, bins, valid) plus setting [n]. fwd is model.make_forward(model, tok)."""
    import jax.numpy as jnp
    from .testset import load_testset
    maze, T = tok.maze, tok.T
    trajs = load_testset() if trajs is None else trajs
    rows = np.arange(len(trajs["length"])) if rows is None else np.asarray(rows)
    pos = trajs["positions"][rows].astype(np.int64)
    L = trajs["length"][rows].astype(np.int64)
    x = tok.with_mode(tok.encode_body(pos, trajs["actions"][rows], L), None)
    v = np.concatenate([np.asarray(fwd(params, jnp.asarray(x[i:i + chunk]))["v_logits"]) for i in range(0, len(rows), chunk)])
    t = np.arange(T + 1)
    true = prefix_expected_return(maze, pos, L)
    model = np.exp(_log_softmax(v)) @ maze.bin_reward
    bins = compute_ground_truth(maze).h[t[None], pos] @ maze.bin_reward
    valid = t[None] <= L[:, None]
    pos_true = valid & (np.nan_to_num(true) > 0)
    log = lambda z: np.log(np.maximum(z, 1e-30))
    err = dict(ev_mae=np.abs(model - true), ev_logerr=np.abs(log(model) - log(true)), ev_bias=log(model) - log(true),
               bins_mae=np.abs(bins - true), bins_logerr=np.abs(log(bins) - log(true)))
    setting = trajs["setting"][rows] if "setting" in trajs else np.zeros(len(rows), dtype=np.int64)
    names = [str(s) for s in trajs["setting_names"]] if "setting_names" in trajs else []
    groups = [(n, setting == i) for i, n in enumerate(names)] + [("all", np.ones(len(rows), dtype=bool))]
    out, per = {}, {}
    for name, sel in groups:
        if not sel.any():
            continue
        per[name] = {k: float(e[(pos_true if "log" in k or "bias" in k else valid) & sel[:, None]].mean())
                     for k, e in err.items()}
        per[name]["n_prefixes"] = int((valid & sel[:, None]).sum())
        for k, val in per[name].items():
            out[f"{k}/{name}"] = val
    out["per_setting"] = per
    if detail:
        out["prefix"] = dict(true=true, model=model, bins=bins, valid=valid, setting=setting)
    return out
