"""Token format. Every sequence slot is a triple (kind, x, y), stored as int32 [..., L, 3]:
    kind   action a (ACT0 + a), POS, a return bin (MODE0 + k), NOR, or PAD
    x, y   column and row of the cell for POS slots; the null index (x = W, y = H) for every other slot
The model embeds a slot as E_kind[kind] + E_x[x] + E_y[y], so cells cost W + H embeddings, not W * H.

Sequence: [MODE] pos_0 a_1 pos_1 ... a_L pos_L PAD ...      L = 2 + 2T; pos_0 is the rollout's start cell.
MODE is NOR (no outcome given) or a return bin k. Step t's state slot (the slot holding pos_t) is
sidx[t] = 2t + 1; the action slot after it is aidx[t] = 2t + 2.

Next-token output vocabulary (one flat softmax, n_out = 4 + n_cells + 1): what can follow a slot.
    a                       an action (follows a state slot: the policy, pi in NOR mode, pi_R in R mode)
    OUT_CELL0 + y * W + x   a cell (follows an action slot: the dynamics P(s' | s, a); after MODE: the start)
    OUT_END                 PAD after the last position (the rollout ended at the goal)
"""
from __future__ import annotations

import numpy as np

from .env import Maze, N_ACTIONS, ACTION_NAMES

TYPE_MODE, TYPE_ACT, TYPE_POS = 0, 1, 2
N_TYPES = 3


class Tokenizer:
    def __init__(self, maze: Maze, cond: str = "bin"):
        """cond: what a reward-bin MODE token k asserts. "bin": the outcome landed in bin k (the original).
        "threshold": the outcome was bin k OR FASTER (arrived within bin k's window) -- nested events, so a row
        that achieved bin b is a valid conditioned sample for every k <= b, and the event's probability is the
        tail sum of the categorical value head. Token ids are identical under both; only the semantics
        (which rows carry which token, and how the value head is read for a query) change."""
        if cond not in ("bin", "threshold"):
            raise ValueError(f"cond={cond!r} not in bin|threshold")
        self.maze, self.cond = maze, cond
        self.T, self.K, self.H, self.W = maze.T, maze.K, maze.H, maze.W
        self.ACT0 = 0
        self.POS = N_ACTIONS
        self.MODE0 = self.POS + 1
        self.NOR = self.MODE0 + self.K
        self.PAD = self.NOR + 1
        self.n_kind = self.PAD + 1
        self.XN, self.YN = self.W, self.H           # null coordinates
        self.n_x, self.n_y = self.W + 1, self.H + 1
        self.L = 2 + 2 * self.T
        self.sidx = 2 * np.arange(self.T + 1) + 1
        self.aidx = 2 * np.arange(self.T) + 2
        self.OUT_ACT0 = 0
        self.OUT_CELL0 = N_ACTIONS
        self.OUT_END = self.OUT_CELL0 + maze.n_cells
        self.n_out = self.OUT_END + 1
        self.types = np.array([TYPE_MODE, TYPE_POS] + [TYPE_ACT, TYPE_POS] * self.T, dtype=np.int32)
        self.pad_slot = np.array([self.PAD, self.XN, self.YN], dtype=np.int32)

    # ---- slots: each returns int32 [..., 3] ----------------------------------------------
    def _slot(self, kind, x=None, y=None) -> np.ndarray:
        kind = np.asarray(kind).astype(np.int32)
        x = np.full_like(kind, self.XN) if x is None else np.asarray(x).astype(np.int32)
        y = np.full_like(kind, self.YN) if y is None else np.asarray(y).astype(np.int32)
        return np.stack(np.broadcast_arrays(kind, x, y), -1)

    def act(self, a):
        return self._slot(self.ACT0 + np.asarray(a).astype(np.int32))

    def pos(self, cell):
        cell = np.asarray(cell).astype(np.int32)
        return self._slot(np.full_like(cell, self.POS), cell % self.W, cell // self.W)

    def mode(self, R_bin=None):
        return self._slot(self.NOR if R_bin is None else self.MODE0 + np.asarray(R_bin).astype(np.int32))

    def blank(self, N: int) -> np.ndarray:
        return np.broadcast_to(self.pad_slot, (N, self.L, 3)).copy()

    # ---- episodes ------------------------------------------------------------------------
    def encode_body(self, positions, actions, length) -> np.ndarray:
        """[N, L, 3] with the MODE slot left as PAD (fill it with with_mode)."""
        tok = self.blank(positions.shape[0])
        valid = (np.arange(self.T)[None, :] < np.asarray(length)[:, None])[..., None]
        tok[:, 1] = self.pos(positions[:, 0])
        tok[:, 2::2] = np.where(valid, self.act(actions), self.pad_slot)
        tok[:, 3::2] = np.where(valid, self.pos(positions[:, 1:]), self.pad_slot)
        return tok

    def with_mode(self, body: np.ndarray, R_bin=None) -> np.ndarray:
        tok = body.copy()
        tok[:, 0] = self.mode(R_bin)
        return tok

    def encode(self, data: dict, idx=None, R_bin=None) -> np.ndarray:
        idx = slice(None) if idx is None else idx
        body = self.encode_body(data["positions"][idx], data["actions"][idx], data["length"][idx])
        return self.with_mode(body, R_bin)

    # ---- next-token targets (teacher forcing) --------------------------------------------
    def out_id(self, slots) -> np.ndarray:
        """Flat next-token id of slots [..., 3]: action a -> a, POS (x, y) -> OUT_CELL0 + y*W + x,
        PAD -> OUT_END. MODE / NOR never follow another slot and map to -1."""
        k, x, y = slots[..., 0], slots[..., 1], slots[..., 2]
        return np.where(k < self.POS, k - self.ACT0,
               np.where(k == self.POS, self.OUT_CELL0 + y * self.W + x,
               np.where(k == self.PAD, self.OUT_END, -1))).astype(np.int32)

    def next_targets(self, tokens: np.ndarray):
        """For sequences [N, L, 3]: targets[:, i] = out_id(tokens[:, i + 1]) and mask[:, i] = slot i is not
        PAD. Both [N, L - 1]. The last position of a rollout that reached the goal gets target OUT_END."""
        return self.out_id(tokens[:, 1:]), tokens[:, :-1, 0] != self.PAD

    # ---- inspection ----------------------------------------------------------------------
    def slot_str(self, s) -> str:
        k, x, y = (int(v) for v in s)
        if k < self.POS:
            return ACTION_NAMES[k - self.ACT0]
        if k == self.POS:
            return f"({x},{y})"
        if k < self.NOR:
            b = k - self.MODE0
            return "R=0" if b == self.maze.FAIL_BIN else f"R~{self.maze.bin_reward[b]:.3g}"
        return "NOR" if k == self.NOR else "PAD"

    def render(self, seq: np.ndarray) -> str:
        """One sequence [L, 3] as text: 'NOR | start (10,8) | L (9,8) | U (9,7) | ... | PAD x366'."""
        parts = [self.slot_str(seq[0]), "start " + self.slot_str(seq[1])]
        for i in range(2, self.L, 2):
            if seq[i, 0] == self.PAD:
                parts.append(f"PAD x{self.L - i}")
                break
            parts.append(f"{self.slot_str(seq[i])} {self.slot_str(seq[i + 1])}")
        return " | ".join(parts)
