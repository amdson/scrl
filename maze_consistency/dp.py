"""Exact ground truth for the random walk on a fixed maze (numpy).

h[t, s, k]           P(outcome bin k | random walk at cell s at time t, goal not yet reached)
child_h[t, s, a, k]  = h[t+1, next(s, a), k]: the V_{t+1}(k | s, a) term of identities (A) and (B)
piR_star[t, s, k, a] = child_h[t, s, a, k] / (4 h[t, s, k]): the exact pi_R (Doob h-transform), nan where h = 0
V_opt, Q_opt         optimal discounted value-to-go E[gamma**tau]: 1 at the goal, 0 at timeout
d[t, s]              random-walk occupancy of rollouts still running at time t (uniform random start)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import Maze, N_ACTIONS


@dataclass
class GroundTruth:
    maze: Maze
    h: np.ndarray          # [T+1, S, K]
    child_h: np.ndarray    # [T, S, 4, K]
    piR_star: np.ndarray   # [T, S, K, 4]
    V_opt: np.ndarray      # [T+1, S]
    Q_opt: np.ndarray      # [T, S, 4]
    d: np.ndarray          # [T+1, S]
    is_goal: np.ndarray    # [S]


def compute_ground_truth(maze: Maze) -> GroundTruth:
    T, K, S, nxt = maze.T, maze.K, maze.n_cells, maze.next_open
    is_goal = np.arange(S) == maze.goal
    h = np.zeros((T + 1, S, K))
    for t in range(T + 1):
        h[t, maze.goal, int(maze.success_bin(t))] = 1.0
    h[T, ~is_goal, maze.FAIL_BIN] = 1.0
    child_h = np.zeros((T, S, N_ACTIONS, K))
    for t in range(T - 1, -1, -1):
        ch = h[t + 1][nxt]                          # [S, 4, K]
        ch[maze.goal] = h[t, maze.goal]             # the goal is absorbing
        child_h[t] = ch
        h[t, ~is_goal] = ch[~is_goal].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        piR = (child_h / N_ACTIONS) / h[:T, :, None, :]
    V_opt = np.zeros((T + 1, S))
    V_opt[:, is_goal] = 1.0
    Q_opt = np.zeros((T, S, N_ACTIONS))
    for t in range(T - 1, -1, -1):
        Q_opt[t] = maze.gamma * V_opt[t + 1][nxt]
        V_opt[t] = np.where(is_goal, 1.0, Q_opt[t].max(-1))
    d = np.zeros((T + 1, S))
    d[0, maze.start_cells] = 1.0 / len(maze.start_cells)
    for t in range(T):
        np.add.at(d[t + 1], nxt.reshape(-1), np.repeat(d[t] * ~is_goal / N_ACTIONS, N_ACTIONS))
    return GroundTruth(maze, h, child_h, np.transpose(piR, (0, 1, 3, 2)), V_opt, Q_opt, d, is_goal)


def check_identities(gt: GroundTruth) -> dict:
    """(B) holds exactly for h, and pi_R* sums to one wherever it is defined."""
    m, ng = gt.maze, ~gt.is_goal
    b_res = max(np.abs(gt.h[t, ng] - gt.child_h[t, ng].mean(1)).max() for t in range(m.T))
    p = gt.piR_star
    ok = np.isfinite(p).all(-1)
    return dict(B_residual=float(b_res), piR_sum_err=float(np.abs(p[ok].sum(-1) - 1).max()),
                p_best=float(gt.h[0, m.start_cells, m.best_bin(m.start_cells)].mean()))
