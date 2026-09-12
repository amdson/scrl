"""The maze: layout, deterministic dynamics, returns and value-head bins, random-walk rollouts.

Cells are flattened row-major: cell = y * W + x (row y counted from the top, column x).
Actions: 0=U 1=D 2=L 3=R. Moving into a wall = stay in place (still costs a step).
Rollouts start at a uniformly random open cell (not the goal); there is no fixed start.
Return: reward 1 on reaching the goal and 0 otherwise, discounted, so R = gamma**L if the goal is reached
in L steps and R = 0 if not, counted from the rollout's own start. Value-head bins (K = n_bins): bin 0 is R = 0; bins 1..K-1 split log R evenly
between log(gamma**T) and 0. Because log R = L log(gamma), each bin is a run of ~T/(K-1) arrival steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WALL, OPEN, START, GOAL = 0, 1, 2, 3
CELL_CHARS = "#.SG"
ACTION_NAMES = "UDLR"
DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
N_ACTIONS = 4


def _next_table(passable: np.ndarray) -> np.ndarray:
    """next_open[cell, a]; moving off the grid or into a wall stays put."""
    H, W = passable.shape
    nxt = np.tile(np.arange(H * W, dtype=np.int32)[:, None], (1, N_ACTIONS))
    for r in range(H):
        for c in range(W):
            if not passable[r, c]:
                continue
            for a, (dr, dc) in enumerate(DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and passable[nr, nc]:
                    nxt[r * W + c, a] = nr * W + nc
    return nxt


def _bfs_dist(nxt: np.ndarray, source: int) -> np.ndarray:
    """Shortest-path distance from every cell to `source` (-1 if unreachable)."""
    dist = np.full(nxt.shape[0], -1, dtype=np.int32)
    dist[source] = 0
    frontier = [source]
    while frontier:
        new = []
        for s in frontier:
            for s2 in nxt[s]:
                if dist[s2] < 0:
                    dist[s2] = dist[s] + 1
                    new.append(int(s2))
        frontier = new
    return dist


@dataclass
class Maze:
    grid: np.ndarray          # int8 [H, W] of WALL / OPEN / GOAL
    goal: int
    T: int                    # horizon (max steps per rollout)
    gamma: float = 0.95
    n_bins: int = 12
    name: str = "maze"

    FAIL_BIN = 0              # value-head bin for R = 0 (goal not reached)

    def __post_init__(self):
        self.grid = np.asarray(self.grid, dtype=np.int8)
        self.H, self.W = self.grid.shape
        self.n_cells = self.H * self.W
        self.K = self.n_bins
        self.next_open = _next_table(self.grid != WALL)
        self.dist = _bfs_dist(self.next_open, self.goal)          # steps to the goal from every cell (-1: wall)
        self.start_cells = np.flatnonzero(self.dist > 0)          # rollouts start uniformly on these
        assert len(self.start_cells) > 0 and self.dist.max() <= self.T

    # ---- returns and bins ----------------------------------------------------------------
    def return_of(self, length, reached):
        return np.where(np.asarray(reached), self.gamma ** np.asarray(length, dtype=np.float64), 0.0)

    def outcome_bin(self, length, reached):
        L = np.asarray(length).astype(np.int64)
        b = 1 + ((self.K - 1) * (self.T - L)) // self.T        # exact integer form of the log-R binning
        return np.where(np.asarray(reached), np.clip(b, 1, self.K - 1), self.FAIL_BIN)

    def success_bin(self, L):
        return self.outcome_bin(L, True)

    @property
    def bin_edges(self) -> np.ndarray:
        """R edges of bins 1..K-1, ascending: K values from gamma**T to 1."""
        return self.gamma ** (self.T * (1 - np.arange(self.K) / (self.K - 1)))

    @property
    def bin_reward(self) -> np.ndarray:
        """Representative R per bin: 0 for bin 0, the geometric centre of the edges otherwise."""
        e = self.bin_edges
        return np.concatenate([[0.0], np.sqrt(e[:-1] * e[1:])])

    def best_bin(self, start):
        """Bin of the optimal outcome from `start`: reaching the goal in dist[start] steps."""
        return self.success_bin(self.dist[np.asarray(start)])

    def R_opt(self, start):
        """Optimal return from `start`: gamma ** dist[start]."""
        return self.gamma ** self.dist[np.asarray(start)].astype(np.float64)

    # ---- display -------------------------------------------------------------------------
    def rc(self, cell):
        return divmod(int(cell), self.W)

    def ascii(self, path=None) -> str:
        rows = []
        for r in range(self.H):
            row = ""
            for c in range(self.W):
                v = self.grid[r, c]
                row += "*" if (path is not None and r * self.W + c in path and v == OPEN) else CELL_CHARS[v]
            rows.append(row)
        return "\n".join(rows)

    def __repr__(self):
        return f"Maze({self.name}: {self.H}x{self.W}, max dist={self.dist.max()}, T={self.T}, K={self.K}, gamma={self.gamma})"


def maze_from_ascii(text: str, T: int, gamma: float = 0.95, n_bins: int = 12, name: str = "maze") -> Maze:
    rows = [ln for ln in text.strip("\n").splitlines() if ln.strip()]
    W = max(len(r) for r in rows)
    grid = np.full((len(rows), W), WALL, dtype=np.int8)
    for r, ln in enumerate(rows):
        for c, ch in enumerate(ln):
            grid[r, c] = CELL_CHARS.index(ch)
    grid[grid == START] = OPEN                                    # no fixed start: an S is just an open cell
    goal = int(np.flatnonzero(grid.reshape(-1) == GOAL)[0])
    return Maze(grid, goal, T, gamma=gamma, n_bins=n_bins, name=name)


def random_walk_episodes(maze: Maze, N: int, seed: int, starts=None) -> dict:
    """N uniform random-walk rollouts of up to T steps, stopping at the goal. Starts are uniform over
    maze.start_cells unless given.

    positions [N, T+1] (frozen after arrival), actions [N, T] (entries after the end are unused),
    length [N] (T if never reached), returns [N] = gamma**length or 0, reached [N].
    """
    rng = np.random.default_rng(seed)
    T = maze.T
    pos = (rng.choice(maze.start_cells, N) if starts is None else np.asarray(starts)).astype(np.int32)
    cum = np.array([0.25, 0.5, 0.75, 1.0])
    positions = np.zeros((N, T + 1), dtype=np.int32)
    positions[:, 0] = pos
    actions = np.zeros((N, T), dtype=np.int8)
    length = np.full(N, T, dtype=np.int32)
    returns = np.zeros(N, dtype=np.float32)
    alive = np.ones(N, dtype=bool)
    for t in range(T):
        a = np.minimum((rng.random(N)[:, None] > cum).sum(-1), N_ACTIONS - 1)
        pos = np.where(alive, maze.next_open[pos, a], pos)
        actions[:, t] = a
        positions[:, t + 1] = pos
        arrived = alive & (pos == maze.goal)
        length[arrived] = t + 1
        returns[arrived] = maze.gamma ** (t + 1)
        alive &= ~arrived
    return dict(positions=positions, actions=actions, length=length, returns=returns, reached=~alive, N=N)
