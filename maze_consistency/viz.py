"""Quality-of-life visualisation: maze rendering, policy arrows, value heatmaps, training curves,
and a one-call per-run report. All functions take a matplotlib Axes where sensible."""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .env import Maze, WALL, START, GOAL, DOOR, DELTAS, ACTION_NAMES

CFG_COLORS = {"MC": "#d62728", "TD": "#1f77b4", "TDA": "#2ca02c", "TDMC": "#9467bd", "FQI": "#ff7f0e",
              "base": "#2ca02c", "c1_rollout": "#1f77b4", "c2_distill": "#ff7f0e", "c3_softq": "#d62728"}


def _grid_rgb(maze: Maze):
    H, W = maze.H, maze.W
    img = np.ones((H, W, 3))
    g = maze.grid
    img[g == WALL] = (0.15, 0.15, 0.15)
    img[g == DOOR] = (0.85, 0.65, 0.2)
    return img


def plot_maze(ax, maze: Maze, cell_values=None, arrows=None, path=None, title=None, cmap="viridis",
              vmin=None, vmax=None, log=False, arrow_color="k", cbar=True, annotate=False):
    """Draw the maze. cell_values: [n_cells] (nan = blank). arrows: [n_cells, 4] action probs."""
    ax.imshow(_grid_rgb(maze), interpolation="nearest")
    if cell_values is not None:
        v = np.asarray(cell_values, dtype=float).reshape(maze.H, maze.W)
        v = np.where(maze.grid == WALL, np.nan, v)
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
        im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, alpha=0.85, interpolation="nearest", norm=norm,
                       vmin=None if log else vmin, vmax=None if log else vmax)
        if cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if annotate:
            for c in range(maze.n_cells):
                if np.isfinite(v.flat[c]):
                    r, cc = maze.rc(c)
                    ax.text(cc, r, f"{v.flat[c]:.2g}", ha="center", va="center", fontsize=6, color="w")
    if arrows is not None:
        arrows = np.asarray(arrows)
        for c in range(maze.n_cells):
            if not np.isfinite(arrows[c]).all():
                continue
            r, cc = maze.rc(c)
            for a in range(4):
                p = arrows[c, a]
                if p < 0.02:
                    continue
                dr, dc = DELTAS[a] * 0.46 * p
                ax.annotate("", xy=(cc + dc, r + dr), xytext=(cc, r),
                            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=0.6 + 2.0 * p, alpha=0.9))
    if path is not None:
        rc = np.array([maze.rc(c) for c in path])
        ax.plot(rc[:, 1], rc[:, 0], "-", color="magenta", lw=2, alpha=0.7)
    sr, sc = maze.rc(maze.start)
    gr, gc = maze.rc(maze.goal)
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold", color="b")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold", color="r")
    if maze.has_door:
        dr, dc = maze.rc(maze.door)
        ax.text(dc, dr, "D", ha="center", va="center", fontsize=10, fontweight="bold", color="k")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_visitation(ax, maze: Maze, positions, length=None, title="visitation"):
    """Heatmap of cells visited by a batch of rollouts (positions [N, T+1])."""
    counts = np.zeros(maze.n_cells)
    for i in range(positions.shape[0]):
        L = int(length[i]) if length is not None else positions.shape[1] - 1
        np.add.at(counts, positions[i, :L + 1], 1)
    counts = np.where(counts > 0, counts / positions.shape[0], np.nan)
    plot_maze(ax, maze, cell_values=counts, title=title, cmap="magma", log=True)


def _curve(ax, x, ys, label, color, ls="-"):
    ys = np.array([[np.nan if v is None else v for v in y] for y in ys], dtype=float)
    mu = np.nanmean(ys, 0)
    ax.plot(x, mu, ls, color=color, label=label, lw=1.8)
    if ys.shape[0] > 1:
        sd = np.nanstd(ys, 0)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15)


def plot_per_t(ax, runs: dict, key="value_err", title=None, ylabel=None, logy=False, which=-1):
    """runs: {label: [result dicts]} -> per-t curve from eval index `which`, mean±sd over runs."""
    for i, (label, results) in enumerate(runs.items()):
        ys = [r["evals"][which][key] for r in results if key in r["evals"][which]]
        if not ys:
            continue
        color = CFG_COLORS.get(label.split("_N")[0].split(" ")[0], f"C{i}")
        _curve(ax, np.arange(len(ys[0])), ys, label, color)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel or key)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_vs_step(ax, runs: dict, key="max_value_err", title=None, logy=False):
    for i, (label, results) in enumerate(runs.items()):
        xs = [e["step"] for e in results[0]["evals"]]
        ys = [[e.get(key, np.nan) if not isinstance(e.get(key), list) else np.nanmax(np.array(e[key], dtype=float)) for e in r["evals"]] for r in results]
        color = CFG_COLORS.get(label.split("_N")[0].split(" ")[0], f"C{i}")
        _curve(ax, xs, ys, label, color)
    ax.set_xlabel("step")
    ax.set_ylabel(key)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_losses(ax, result, keys=("pi", "piR", "mc", "td", "term", "a", "fqi"), title="losses"):
    h = result["history"]
    if not h:
        return
    xs = [r["step"] for r in h]
    for k in keys:
        if k in h[0]:
            ax.plot(xs, [r[k] for r in h], label=k, lw=1.2)
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def run_report(run_dir: str, maze: Maze | None = None, gt=None, save=True):
    """Multi-panel figure for one run: losses, value_err[t] over evals, opt/consistency vs step,
    P(R=-d* | t, s) model vs DP at a few t, and pi_R(R=-d*) arrows model vs DP."""
    from .env import get_maze
    from .dp import compute_ground_truth
    with open(os.path.join(run_dir, "metrics.json")) as f:
        res = json.load(f)
    maze = maze or get_maze(res["config"]["maze"])
    gt = gt or compute_ground_truth(maze)
    det = None
    p = os.path.join(run_dir, "detail.npz")
    if os.path.exists(p):
        det = np.load(p, allow_pickle=True)
    ev = res["evals"]
    n_t = 3
    fig, axes = plt.subplots(2 + (2 if det is not None else 0), 3, figsize=(13, 3.6 * (2 + (2 if det is not None else 0))))
    ax = axes.reshape(-1)
    plot_losses(ax[0], res)
    for i, e in enumerate(ev):
        ax[1].plot(np.array(e["value_err"], dtype=float), color=plt.cm.viridis(i / max(len(ev) - 1, 1)), lw=1.2,
                   label=f"step {e['step']}" if i in (0, len(ev) - 1) else None)
    ax[1].set_title("value_err[t] (TV to DP) over training", fontsize=9); ax[1].set_xlabel("t"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    xs = [e["step"] for e in ev]
    for k in ("opt_err", "mean_consistency", "mean_piR_err", "max_value_err"):
        ax[2].plot(xs, [e[k] for e in ev], label=k, lw=1.4)
    ax[2].set_yscale("log"); ax[2].legend(fontsize=7); ax[2].set_title("scalar metrics vs step", fontsize=9); ax[2].grid(alpha=0.3)
    ax[3].plot(np.array(ev[-1]["consistency"], dtype=float), label="(A) residual"); ax[3].set_title("consistency[t] (final)", fontsize=9); ax[3].set_xlabel("t"); ax[3].grid(alpha=0.3)
    ax[4].plot(np.array(ev[-1]["logp_rstar_err"], dtype=float)); ax[4].set_title("|log V_t(R*) - log h_t(R*)| (final)", fontsize=9); ax[4].set_xlabel("t"); ax[4].grid(alpha=0.3)
    if "rollouts" in ev[-1]:
        names = list(ev[-1]["rollouts"])
        sr = [ev[-1]["rollouts"][k]["solve_rate"] for k in names]
        orr = [ev[-1]["rollouts"][k]["optimal_rate"] for k in names]
        x = np.arange(len(names))
        ax[5].bar(x - 0.2, sr, 0.4, label="solve"); ax[5].bar(x + 0.2, orr, 0.4, label="optimal")
        ax[5].set_xticks(x); ax[5].set_xticklabels(names, rotation=60, fontsize=7); ax[5].legend(fontsize=7); ax[5].set_title("decode rollouts", fontsize=9)
    if det is not None:
        ts = list(det["t_list"])
        pick = [ts[0], ts[len(ts) // 2], ts[-1]]
        for j, t in enumerate(pick):
            i = ts.index(t)
            vt = np.array(det["p_rstar_true"][i], dtype=float)
            vm = np.array(det["p_rstar_model"][i], dtype=float)
            vmax = np.nanmax(vt) if np.nanmax(vt) > 0 else 1
            plot_maze(ax[6 + j], maze, cell_values=np.where(vt > 0, vt, np.nan), title=f"DP P(R=-d*|t={t},s)", log=True, vmin=1e-5, vmax=vmax)
            plot_maze(ax[9 + j], maze, cell_values=np.where(vm > 1e-6, vm, np.nan), title=f"model P(R=-d*|t={t},s)", log=True, vmin=1e-5, vmax=vmax)
    fig.suptitle(f"{res['config']['exp']}/{res['config']['name']}  loss={res['config']['loss']} N={res['data']['N']}", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(run_dir, "report.png"), dpi=120)
    # separate arrow figure
    if det is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
        plot_maze(ax2[0], maze, arrows=det["piR_star_true"], title="DP pi_R*(a | s, R=-d*) on shortest paths")
        plot_maze(ax2[1], maze, arrows=det["piR_star_model"], title="model pi_R(a | s, R=-d*)", arrow_color="darkgreen")
        fig2.tight_layout()
        if save:
            fig2.savefig(os.path.join(run_dir, "piR_arrows.png"), dpi=120)
        return fig, fig2
    return fig, None


def shortest_path_arrows(maze: Maze, gt):
    """pi_R*(.|s, R=-d*) evaluated at t = dist(start, s): defined exactly on shortest-path cells."""
    from .env import _bfs_dist
    dist = _bfs_dist(maze.next_open, maze.start)
    k = int(maze.bin_of(maze.R_max))
    arr = np.full((maze.n_cells, 4), np.nan)
    for c in range(maze.n_cells):
        t = int(dist[c])
        if 0 <= t < maze.T:
            arr[c] = gt.piR_star[t, c, k]
    return arr


def plot_ground_truth(maze: Maze, gt, save_path=None):
    """Overview of the DP truth: P(reach), P(R=-d*) at t=0, h-transform arrows at R=-d*, Q_opt greedy."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    plot_maze(ax[0], maze, cell_values=1 - gt.h[0, :maze.n_cells, 0], title="P(reach goal | t=0, s) random walk")
    v = gt.h[0, :maze.n_cells, maze.bin_of(maze.R_max)]
    plot_maze(ax[1], maze, cell_values=np.where(v > 0, v, np.nan), title="P(R=-d* | t=0, s)", log=True)
    plot_maze(ax[2], maze, arrows=shortest_path_arrows(maze, gt), title="pi_R*(a|s,R=-d*) at t=dist(S,s) (h-transform)")
    plot_maze(ax[3], maze, arrows=gt.policy_greedy_opt()[0, :maze.n_cells], cell_values=gt.V_opt[0, :maze.n_cells], title="Q_opt greedy, V_opt (t=0)", cmap="cividis")
    fig.suptitle(repr(maze), fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def print_rollout_table(rollouts: dict, title=""):
    print(title)
    print(f"{'policy':18s} {'solve':>6s} {'optimal':>8s} {'steps':>6s} {'meanR':>7s} {'door':>6s}")
    for k, v in rollouts.items():
        door = "" if v.get("door_choice") is None else f"{v['door_choice']:.2f}"
        print(f"{k:18s} {v['solve_rate']:6.2f} {v['optimal_rate']:8.3f} {v['mean_steps']:6.1f} {v['mean_return']:7.2f} {door:>6s}")
