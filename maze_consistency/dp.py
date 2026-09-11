"""Ground truth by dynamic programming (numpy).

h[t, s, k]        P(episode return = R_k | at DP-state s at time t, not yet done) under the random walk.
child_h[t,s,a,k]  = V_t(R_k | s, a) = sum_j tprob[s,a,j] h[t+1, trans[s,a,j]]   (the (A)/(B) child term)
piR_star[t,s,k,a] = pi(a) child_h[t,s,a,k] / h[t,s,k]           (Doob h-transform; nan where h=0)
Q_rw[t,s,a]       = E[R | s, a, t] under the random walk (total return)
V_opt / Q_opt     return-to-go under the optimal (max-backup) policy; horizon-aware
d[t, s]           random-walk occupancy of alive episodes
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .env import Maze, FLAG_UNKNOWN, FLAG_OPEN, FLAG_LOCKED, N_ACTIONS


def build_transitions(maze: Maze):
    """Dense transition model over DP states. Returns (trans [S,4,2] int, tprob [S,4,2] float)."""
    n, F = maze.n_cells, maze.n_dp_flags
    S = n * F
    trans = np.zeros((S, N_ACTIONS, 2), dtype=np.int64)
    tprob = np.zeros((S, N_ACTIONS, 2), dtype=np.float64)
    for f in range(F):
        for pos in range(n):
            s = pos + n * f
            for a in range(N_ACTIONS):
                if not maze.has_door:
                    trans[s, a, 0] = maze.next_open[pos, a]
                    tprob[s, a, 0] = 1.0
                    continue
                nxt_open = maze.next_open[pos, a]
                if f == FLAG_UNKNOWN and nxt_open == maze.door and pos != maze.door:
                    trans[s, a, 0] = pos + n * FLAG_LOCKED            # bumped: stay, learn locked
                    tprob[s, a, 0] = maze.p_locked
                    trans[s, a, 1] = maze.door + n * FLAG_OPEN         # entered: learn open
                    tprob[s, a, 1] = 1.0 - maze.p_locked
                elif f == FLAG_LOCKED:
                    trans[s, a, 0] = maze.next_locked[pos, a] + n * f
                    tprob[s, a, 0] = 1.0
                else:
                    trans[s, a, 0] = nxt_open + n * f
                    tprob[s, a, 0] = 1.0
    return trans, tprob


@dataclass
class GroundTruth:
    maze: Maze
    trans: np.ndarray
    tprob: np.ndarray
    h: np.ndarray          # [T+1, S, K]
    child_h: np.ndarray    # [T, S, 4, K]
    piR_star: np.ndarray   # [T, S, K, 4]  (nan where undefined)
    Q_rw: np.ndarray       # [T, S, 4]
    V_opt: np.ndarray      # [T+1, S]   return-to-go
    Q_opt: np.ndarray      # [T, S, 4]
    d: np.ndarray          # [T+1, S]   occupancy of alive episodes under the random walk
    is_goal: np.ndarray    # [S] bool

    @property
    def S(self):
        return self.h.shape[1]

    def start_state(self):
        return self.maze.start  # flag unknown = 0

    # ---- reference decode policies as [T, S, 4] tables --------------------
    def policy_piR(self, R: int) -> np.ndarray:
        k = int(self.maze.bin_of(R))
        p = self.piR_star[:, :, k, :].copy()
        bad = ~np.isfinite(p).all(-1)
        p[bad] = 1.0 / N_ACTIONS
        return p

    def policy_posterior_tilt(self, R: int, eps=1e-12) -> np.ndarray:
        """pi(a) V_t(R | s, a), normalised; uniform fallback where R is unreachable."""
        k = int(self.maze.bin_of(R))
        w = self.child_h[:, :, :, k] / N_ACTIONS
        z = w.sum(-1, keepdims=True)
        return np.where(z > eps, w / np.maximum(z, eps), 1.0 / N_ACTIONS)

    def policy_ev_tilt(self, beta: float) -> np.ndarray:
        """pi(a) exp(beta * E[R | s, a]) normalised, with E under the random walk."""
        q = self.Q_rw
        logits = beta * (q - q.max(-1, keepdims=True))
        p = np.exp(logits)
        return p / p.sum(-1, keepdims=True)

    def policy_greedy_opt(self) -> np.ndarray:
        best = self.Q_opt >= self.Q_opt.max(-1, keepdims=True) - 1e-9
        return best / best.sum(-1, keepdims=True)

    def expected_return(self, s_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        return (self.h[t_idx, s_idx] * self.maze.R_values).sum(-1)


def compute_ground_truth(maze: Maze) -> GroundTruth:
    T, K, n = maze.T, maze.K, maze.n_cells
    trans, tprob = build_transitions(maze)
    S = trans.shape[0]
    pos_of = np.arange(S) % n
    is_goal = pos_of == maze.goal

    h = np.zeros((T + 1, S, K))
    child_h = np.zeros((T, S, N_ACTIONS, K))
    # base cases
    for t in range(T + 1):
        h[t, is_goal, maze.bin_of(-t)] = 1.0
    h[T, ~is_goal, maze.bin_of(-(T + 1))] = 1.0
    for t in range(T - 1, -1, -1):
        nxt = h[t + 1][trans]                              # [S,4,2,K]
        ch = (tprob[..., None] * nxt).sum(2)               # [S,4,K]
        ch[is_goal] = h[t, is_goal][:, None, :]            # goal is absorbing
        child_h[t] = ch
        h[t, ~is_goal] = ch[~is_goal].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        piR = (child_h / N_ACTIONS) / h[:T][:, :, None, :]   # [T,S,4,K]
    piR = np.transpose(piR, (0, 1, 3, 2))                    # [T,S,K,4]
    Q_rw = (child_h * maze.R_values).sum(-1)                 # [T,S,4]

    V_opt = np.zeros((T + 1, S))
    V_opt[T, ~is_goal] = -1.0
    Q_opt = np.zeros((T, S, N_ACTIONS))
    for t in range(T - 1, -1, -1):
        Q_opt[t] = -1.0 + (tprob * V_opt[t + 1][trans]).sum(-1)
        V_opt[t] = np.where(is_goal, 0.0, Q_opt[t].max(-1))

    d = np.zeros((T + 1, S))
    d[0, maze.start] = 1.0
    for t in range(T):
        alive = d[t] * (~is_goal)
        for j in range(2):
            np.add.at(d[t + 1], trans[:, :, j].reshape(-1),
                      (alive[:, None] * tprob[:, :, j] / N_ACTIONS).reshape(-1))
    return GroundTruth(maze, trans, tprob, h, child_h, piR, Q_rw, V_opt, Q_opt, d, is_goal)


def check_identities(gt: GroundTruth, tol=1e-9) -> dict:
    """Sanity: (B) holds exactly for h, and pi_R* sums to one where defined."""
    m = gt.maze
    T = m.T
    b_res = 0.0
    for t in range(T):
        lhs = gt.h[t, ~gt.is_goal]
        rhs = gt.child_h[t, ~gt.is_goal].mean(1)
        b_res = max(b_res, np.abs(lhs - rhs).max())
    p = gt.piR_star
    ok = np.isfinite(p).all(-1)
    sums = p[ok].sum(-1)
    return dict(B_residual=float(b_res), piR_sum_err=float(np.abs(sums - 1).max()),
                B_ok=b_res < tol, h0_Rmax=float(gt.h[0, m.start, m.bin_of(m.R_max)]))
