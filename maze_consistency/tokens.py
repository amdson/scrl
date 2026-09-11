"""Token encoding for episodes.

Sequence layout (grid tokens optional, off by default for a fixed maze):
    [MODE] [grid x n_cells]? [a_1 pos_1] [a_2 pos_2] ... [a_L pos_L] [PAD ...]
Step t's *state token* (the token at which the model reads V_t and predicts a_{t+1}) is
    sidx[t] = n_grid + 2 t          (t = 0 -> the MODE token / last grid token)
pos tokens encode (cell, known_locked) for the door variant: POS0 + cell + n_cells * known_locked.
"""
from __future__ import annotations

import numpy as np

from .env import Maze, FLAG_LOCKED, N_ACTIONS

TYPE_MODE, TYPE_ACT, TYPE_POS, TYPE_PAD, TYPE_GRID = 0, 1, 2, 3, 4
N_TYPES = 5


class Tokenizer:
    def __init__(self, maze: Maze, use_grid: bool = False):
        self.maze = maze
        self.T, self.K = maze.T, maze.K
        self.n_cells = maze.n_cells
        self.n_flags = maze.n_token_flags
        self.ACT0 = 0
        self.POS0 = N_ACTIONS
        self.MODE0 = self.POS0 + self.n_cells * self.n_flags
        self.NOR = self.MODE0 + self.K
        self.PAD = self.NOR + 1
        self.GRID0 = self.PAD + 1
        self.vocab = self.GRID0 + 5
        self.use_grid = use_grid
        self.n_grid = self.n_cells if use_grid else 0
        self.L = 1 + self.n_grid + 2 * self.T
        self.sidx = self.n_grid + 2 * np.arange(self.T + 1)
        types = np.full(self.L, TYPE_PAD, dtype=np.int32)
        types[0] = TYPE_MODE
        types[1:1 + self.n_grid] = TYPE_GRID
        types[self.n_grid + 1::2] = TYPE_ACT
        types[self.n_grid + 2::2] = TYPE_POS
        self.types = types
        self.grid_tokens = self.GRID0 + maze.grid.reshape(-1).astype(np.int32)

    def mode_token(self, R_bin=None):
        return self.NOR if R_bin is None else self.MODE0 + np.asarray(R_bin)

    def pos_token(self, cell, flag):
        known_locked = (np.asarray(flag) == FLAG_LOCKED).astype(np.int32) if self.n_flags == 2 else 0
        return self.POS0 + np.asarray(cell) + self.n_cells * known_locked

    def encode_body(self, positions, flags, actions, length) -> np.ndarray:
        """Tokens for a batch of episodes with the MODE slot left as PAD (fill it via set_mode)."""
        N = positions.shape[0]
        tok = np.full((N, self.L), self.PAD, dtype=np.int32)
        if self.use_grid:
            tok[:, 1:1 + self.n_grid] = self.grid_tokens[None]
        steps = np.arange(self.T)[None, :]
        valid = steps < length[:, None]
        act_tok = self.ACT0 + actions.astype(np.int32)
        pos_tok = self.pos_token(positions[:, 1:], flags[:, 1:])
        body_a = np.where(valid, act_tok, self.PAD)
        body_p = np.where(valid, pos_tok, self.PAD)
        tok[:, self.n_grid + 1::2] = body_a
        tok[:, self.n_grid + 2::2] = body_p
        return tok

    def with_mode(self, body: np.ndarray, R_bin=None) -> np.ndarray:
        tok = body.copy()
        tok[:, 0] = self.mode_token(R_bin)
        return tok

    def encode(self, data: dict, idx=None, R_bin=None) -> np.ndarray:
        idx = slice(None) if idx is None else idx
        body = self.encode_body(data["positions"][idx], data["flags"][idx], data["actions"][idx], data["length"][idx])
        return self.with_mode(body, R_bin)


def r_distribution(maze: Maze, top_weight: float = 3.0, n_top: int = 5) -> np.ndarray:
    """r(R) for the (A) loss: uniform over achievable bins [-(T+1), -d*] with extra weight near -d*."""
    K = maze.K
    p = np.zeros(K)
    kmax = int(maze.bin_of(maze.R_max))
    p[: kmax + 1] = 1.0
    p[max(0, kmax - n_top + 1): kmax + 1] = top_weight
    return p / p.sum()
