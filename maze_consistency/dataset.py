"""The canonical dataset: one fixed 12x12 maze, 100k deterministic random-walk rollouts of up to T = 200
steps. Each rollout starts at a uniformly random open cell and stops at the goal.

Rollout record (row-aligned on the rollout index i):
    positions  int16 [N, T+1]   cell at t = 0..T; positions[:, 0] is the start; frozen after the goal
    actions    int8  [N, T]     action at t (0=U 1=D 2=L 3=R); entries after the rollout ends are unused
    length     int16 [N]        L = steps to reach the goal, or T if never reached
    returns    float32 [N]      R = GAMMA**L if reached, else 0, counted from the rollout's own start
    reached    bool  [N]
The value head's classes are maze.outcome_bin(length, reached): 0 = R = 0, then N_BINS - 1 log-spaced bins.
Moving into a wall = stay in place (still costs a step).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .env import Maze, maze_from_ascii, random_walk_episodes, WALL, N_ACTIONS
from .dp import compute_ground_truth

DATA_DIR = "data/canonical"
MAZE_FILE = os.path.join(DATA_DIR, "maze.txt")   # source of truth for the layout; hand-edited, never overwritten
T_MAX = 200
GAMMA = 0.95
N_BINS = 12
N_ROLLOUTS = 100_000
DATA_SEED = 2026


def canonical_maze(path: str = MAZE_FILE) -> Maze:
    with open(path) as f:
        return maze_from_ascii(f.read(), T=T_MAX, name="canon12", gamma=GAMMA, n_bins=N_BINS)


def build(out_dir: str = DATA_DIR, N: int = N_ROLLOUTS, seed: int = DATA_SEED, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    maze = canonical_maze()
    log(repr(maze))
    log(maze.ascii())
    data = random_walk_episodes(maze, N, seed)
    rec = dict(positions=data["positions"].astype(np.int16), actions=data["actions"].astype(np.int8),
               length=data["length"].astype(np.int16), returns=data["returns"].astype(np.float32),
               reached=data["reached"])
    np.savez_compressed(os.path.join(out_dir, "rollouts.npz"), **rec)
    np.save(os.path.join(out_dir, "grid.npy"), maze.grid)
    stats = dataset_stats(maze, rec)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(maze_file=MAZE_FILE, maze_ascii=maze.ascii().splitlines(), N=N, seed=seed, goal=maze.goal,
                       goal_xy=[maze.goal % maze.W, maze.goal // maze.W], start_rule="uniform over open non-goal cells",
                       n_start_cells=int(len(maze.start_cells)), max_dist=int(maze.dist.max()), T=maze.T, K=maze.K,
                       H=maze.H, W=maze.W, gamma=maze.gamma, n_bins=maze.K, bin_edges=maze.bin_edges.tolist(),
                       reward_rule="R = gamma**length (from the rollout's start) if reached else 0"), f, indent=1)
    plot_dataset(maze, rec, stats, os.path.join(out_dir, "dataset.png"))
    return dict(maze=maze, data=rec, stats=stats)


def load(out_dir: str = DATA_DIR):
    maze = canonical_maze()
    z = np.load(os.path.join(out_dir, "rollouts.npz"))
    return maze, {k: z[k] for k in z.files}


def shortest_path_counts(maze: Maze) -> np.ndarray:
    """Number of distinct shortest paths from each cell to the goal (0 for walls)."""
    ways = np.zeros(maze.n_cells, dtype=np.float64)
    ways[maze.goal] = 1
    for dd in range(1, int(maze.dist.max()) + 1):
        for c in np.flatnonzero(maze.dist == dd):
            ways[c] = sum(ways[s2] for s2 in maze.next_open[c] if maze.dist[s2] == dd - 1)
    return ways


def dataset_stats(maze: Maze, rec: dict) -> dict:
    T, N, K = maze.T, len(rec["length"]), maze.K
    gt = compute_ground_truth(maze)
    reached, L = rec["reached"], rec["length"].astype(np.int64)
    s0 = rec["positions"][:, 0].astype(np.int64)
    d0 = maze.dist[s0]
    bins = maze.outcome_bin(L, reached)
    hist = np.bincount(bins, minlength=K)
    h0 = gt.h[0, s0]                                               # exact outcome distribution for each rollout's start
    own_best = maze.best_bin(s0)
    is_best = bins == own_best
    is_opt = reached & (L == d0)
    p_best = h0[np.arange(N), own_best]
    p_opt = shortest_path_counts(maze)[s0] / 4.0 ** d0
    alive = np.arange(T + 1)[None, :] <= L[:, None]
    occ = np.zeros(maze.n_cells)
    np.add.at(occ, rec["positions"][alive].astype(np.int64), 1.0)
    occ /= alive.sum()
    ts = np.zeros((T + 1, maze.n_cells), dtype=np.int64)
    tt = np.broadcast_to(np.arange(T + 1)[None, :], rec["positions"].shape)[alive]
    np.add.at(ts, (tt, rec["positions"][alive].astype(np.int64)), 1)
    reachable = gt.d > 0
    by_dist = []
    for dd in range(1, int(maze.dist.max()) + 1):
        sel = d0 == dd
        if not sel.any():
            continue
        by_dist.append(dict(dist=dd, n_cells=int((maze.dist == dd).sum()), rollouts=int(sel.sum()),
                            reach=float(reached[sel].mean()), best_bin=int(maze.success_bin(dd)),
                            best_bin_count=int(is_best[sel].sum()), best_bin_expected=float(p_best[sel].sum()),
                            optimal_count=int(is_opt[sel].sum()), optimal_expected=float(p_opt[sel].sum())))
    Lr = L[reached]
    return dict(
        N=int(N), T=int(T), gamma=float(maze.gamma), n_bins=int(K), n_start_cells=int(len(maze.start_cells)),
        max_dist=int(maze.dist.max()),
        reach_rate=float(reached.mean()), reach_rate_dp=float(1 - h0[:, maze.FAIL_BIN].mean()),
        best_bin_rate=float(is_best.mean()), best_bin_rate_dp=float(p_best.mean()),
        optimal_rate=float(is_opt.mean()), optimal_rate_dp=float(p_opt.mean()),
        length_reached=dict(mean=float(Lr.mean()), median=float(np.median(Lr)), min=int(Lr.min())),
        return_mean=float(rec["returns"].mean()),
        return_hist=hist.tolist(), return_hist_dp=h0.sum(0).tolist(),
        bins=_bin_table(maze, hist, h0.sum(0)),
        by_distance=by_dist,
        cells_visited=int((occ > 0).sum()),
        ts_coverage=float((ts[reachable] > 0).mean()), ts_pairs_reachable=int(reachable.sum()),
        occupancy=occ.tolist(),
        transitions=int(alive[:, :T].sum()),
        wall_bump_rate=float((rec["positions"][:, 1:] == rec["positions"][:, :-1])[alive[:, :T]].mean()),
    )


def _bin_table(maze: Maze, hist, expected) -> list:
    Ls = np.arange(maze.T + 1)
    bl = maze.success_bin(Ls)
    rows = []
    for k in range(maze.K):
        if k == maze.FAIL_BIN:
            label, L_range = "0: R=0", None
        else:
            lo, hi = int(Ls[bl == k].min()), int(Ls[bl == k].max())
            label, L_range = f"{k}: L{lo}-{hi}", [lo, hi]
        rows.append(dict(bin=k, label=label, L_range=L_range, R_center=float(maze.bin_reward[k]),
                         count=int(hist[k]), expected_dp=float(expected[k])))
    return rows


def plot_dataset(maze: Maze, rec: dict, stats: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .viz import plot_maze
    T = maze.T
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    dist = np.where(maze.dist >= 0, maze.dist, np.nan).astype(float)
    plot_maze(ax[0, 0], maze, cell_values=dist, title=f"maze ({maze.H}x{maze.W}), steps to goal", cmap="cividis", annotate=True)
    occ = np.array(stats["occupancy"])
    plot_maze(ax[0, 1], maze, cell_values=np.where(occ > 0, occ, np.nan), title="random-walk occupancy (log)", cmap="magma", log=True)
    L, s0 = rec["length"].astype(int), rec["positions"][:, 0].astype(int)
    far = np.flatnonzero(rec["reached"] & (maze.dist[s0] >= np.percentile(maze.dist[s0], 75)))
    i = int(far[np.argmin(L[far])]) if len(far) else int(np.flatnonzero(rec["reached"])[0])
    plot_maze(ax[0, 2], maze, path=rec["positions"][i, :L[i] + 1],
              title=f"fastest rollout from a far start (dist {maze.dist[s0[i]]}, L={L[i]})")
    h, hd, ks = np.array(stats["return_hist"], float), np.array(stats["return_hist_dp"], float), np.arange(maze.K)
    ax[1, 0].bar(ks, h, label="data")
    ax[1, 0].plot(ks, hd, "k.-", lw=1, label="DP expected")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_xticks(ks)
    ax[1, 0].set_xticklabels([b["label"] for b in stats["bins"]], rotation=60, fontsize=7)
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)
    ax[1, 0].set_title(f"value-head bins (gamma={maze.gamma}), all starts", fontsize=9)
    bd = stats["by_distance"]
    x = np.array([r["dist"] for r in bd])
    ax[1, 1].bar(x - 0.2, [max(r["best_bin_count"], 0.5) for r in bd], 0.4, label="reached own best bin")
    ax[1, 1].bar(x + 0.2, [max(r["optimal_count"], 0.5) for r in bd], 0.4, label="exactly optimal")
    ax[1, 1].plot(x - 0.2, [r["best_bin_expected"] for r in bd], "k.-", lw=1, label="DP expected")
    ax[1, 1].plot(x + 0.2, [r["optimal_expected"] for r in bd], "k.:", lw=1)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("start distance to goal"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3); ax[1, 1].set_title("rollouts reaching their best outcome, by start distance", fontsize=9)
    cum = np.array([(rec["reached"] & (L <= t)).mean() for t in range(T + 1)])
    ax[1, 2].plot(cum); ax[1, 2].set_xlabel("t"); ax[1, 2].set_title("P(reached by t)", fontsize=9); ax[1, 2].grid(alpha=0.3)
    fig.suptitle(f"canonical dataset: N={stats['N']:,}, T={T}, uniform random starts, {stats['transitions']:,} transitions, "
                 f"reached {stats['reach_rate']:.1%}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig
