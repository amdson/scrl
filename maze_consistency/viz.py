"""Maze rendering: walls, start and goal, per-cell heatmaps, policy arrows, paths."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .env import Maze, WALL, DELTAS


def plot_maze(ax, maze: Maze, cell_values=None, arrows=None, path=None, title=None, cmap="viridis",
              vmin=None, vmax=None, log=False, arrow_color="k", cbar=True, annotate=False):
    """cell_values: [n_cells] (nan = blank). arrows: [n_cells, 4] action probabilities. path: cell list."""
    img = np.ones((maze.H, maze.W, 3))
    img[maze.grid == WALL] = (0.15, 0.15, 0.15)
    ax.imshow(img, interpolation="nearest")
    if cell_values is not None:
        v = np.asarray(cell_values, dtype=float).reshape(maze.H, maze.W)
        v = np.where(maze.grid == WALL, np.nan, v)
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
        im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, alpha=0.85, interpolation="nearest", norm=norm,
                       vmin=None if log else vmin, vmax=None if log else vmax)
        if cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if annotate:
            for c in np.flatnonzero(np.isfinite(v.reshape(-1))):
                r, cc = maze.rc(c)
                ax.text(cc, r, f"{v.flat[c]:.2g}", ha="center", va="center", fontsize=6, color="w")
    if arrows is not None:
        for c in range(maze.n_cells):
            if not np.isfinite(arrows[c]).all():
                continue
            r, cc = maze.rc(c)
            for a in range(4):
                p = arrows[c, a]
                if p >= 0.02:
                    dr, dc = DELTAS[a] * 0.46 * p
                    ax.annotate("", xy=(cc + dc, r + dr), xytext=(cc, r),
                                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=0.6 + 2.0 * p))
    if path is not None:
        rc = np.array([maze.rc(c) for c in path])
        ax.plot(rc[:, 1], rc[:, 0], "-", color="magenta", lw=2, alpha=0.7)
        ax.plot(rc[0, 1], rc[0, 0], "o", color="magenta", ms=9)             # the rollout's start
    r, c = maze.rc(maze.goal)
    ax.text(c, r, "G", ha="center", va="center", fontsize=12, fontweight="bold", color="r")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def optimal_arrows(maze: Maze, gt) -> np.ndarray:
    """Exact pi_R(. | start s, s's own best bin) at t = 0, for every start cell."""
    arr = np.full((maze.n_cells, 4), np.nan)
    arr[maze.start_cells] = gt.piR_star[0, maze.start_cells, maze.best_bin(maze.start_cells)]
    return arr


def plot_ground_truth(maze: Maze, gt, save_path=None):
    """Per start cell: P(reach), P(own best bin), exact pi_R at the own best bin, optimal value."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    n = maze.n_cells
    plot_maze(ax[0], maze, cell_values=1 - gt.h[0, :, maze.FAIL_BIN], title="P(reach goal | t=0, s), random walk")
    v = np.full(n, np.nan)
    v[maze.start_cells] = gt.h[0, maze.start_cells, maze.best_bin(maze.start_cells)]
    plot_maze(ax[1], maze, cell_values=v, log=True, title="P(own best bin | start s)")
    plot_maze(ax[2], maze, arrows=optimal_arrows(maze, gt), title="exact pi_R(a | start s, own best bin)")
    q = gt.Q_opt[0]
    greedy = (q >= q.max(-1, keepdims=True) - 1e-12).astype(float)
    plot_maze(ax[3], maze, cell_values=gt.V_opt[0], arrows=greedy / greedy.sum(-1, keepdims=True),
              cmap="cividis", title="optimal value E[gamma**tau], greedy actions")
    fig.suptitle(repr(maze), fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
