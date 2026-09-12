#!/usr/bin/env python
"""CLI.
  python run.py dataset      # build the canonical dataset from data/canonical/maze.txt -> data/canonical/
  python run.py truth        # exact random-walk ground truth figure -> data/canonical/truth.png
  python run.py show [i]     # rollout i (default: the shortest) drawn on the maze and as tokens
  python run.py testset      # exact test set from the DP -> data/canonical/testset.npz
  python run.py train [name] [steps] [consistency]   # half NOR half R mode; optional TD + A losses
  python run.py eval [name]            # accuracy by MODE setting and in-maze return by requested bin
  python run.py sweep [steps]          # train base / mc / td / mc_a4 / td_a4 with exact-test tracking -> runs/sweep/
  python run.py sweep-plot             # test-metric curves for the sweep -> runs/sweep/curves.png
"""
import sys

import numpy as np


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dataset":
        from maze_consistency import dataset
        s = dataset.build()["stats"]
        print(f"N={s['N']:,} T={s['T']} start cells={s['n_start_cells']} (uniform) max distance to goal={s['max_dist']}")
        print(f"return: R = {s['gamma']}**L from the rollout's start if reached, else 0; {s['n_bins']} bins")
        print(f"reached {s['reach_rate']:.2%} (DP {s['reach_rate_dp']:.2%}); own best bin {s['best_bin_rate']:.2%} "
              f"(DP {s['best_bin_rate_dp']:.2%}); exactly optimal {s['optimal_rate']:.3%} (DP {s['optimal_rate_dp']:.3%})")
        print("bins (label, count, DP expected):", [(b["label"], b["count"], round(b["expected_dp"], 1)) for b in s["bins"]])
        print("by start distance: dist, rollouts, reach, own-best-bin count / DP expected, exactly-optimal count / DP expected")
        for r in s["by_distance"]:
            print(f"  {r['dist']:2d} {r['rollouts']:6d} {r['reach']:.2f}  bin {r['best_bin']:2d}: {r['best_bin_count']:6d} / {r['best_bin_expected']:8.1f}"
                  f"   optimal: {r['optimal_count']:5d} / {r['optimal_expected']:8.1f}")
    elif cmd == "truth":
        from maze_consistency.dataset import canonical_maze
        from maze_consistency.dp import compute_ground_truth, check_identities
        from maze_consistency.viz import plot_ground_truth
        m = canonical_maze()
        gt = compute_ground_truth(m)
        print(m, check_identities(gt))
        plot_ground_truth(m, gt, "data/canonical/truth.png")
    elif cmd == "show":
        from maze_consistency.dataset import load
        from maze_consistency.tokens import Tokenizer
        maze, d = load()
        if len(sys.argv) > 2:
            i = int(sys.argv[2])
        else:
            i = int(np.flatnonzero(d["reached"])[np.argmin(d["length"][d["reached"]])])
        L = int(d["length"][i])
        print(f"rollout {i}: length {L}, return {float(d['returns'][i]):.4g}, reached {bool(d['reached'][i])}")
        print(maze.ascii(path=set(int(c) for c in d["positions"][i, :L + 1])))
        tok = Tokenizer(maze)
        print(tok.render(tok.encode(d, idx=np.array([i]))[0]))
    elif cmd == "testset":
        from maze_consistency.dataset import canonical_maze
        from maze_consistency.testset import build_testset
        build_testset(canonical_maze())
    elif cmd == "train":
        from maze_consistency.train import train
        name = sys.argv[2] if len(sys.argv) > 2 else "tf"
        train(name=name, steps=int(sys.argv[3]) if len(sys.argv) > 3 else 2000,
              consistency=len(sys.argv) > 4 and sys.argv[4] == "consistency")
    elif cmd == "eval":
        from maze_consistency.dataset import load
        from maze_consistency.tokens import Tokenizer
        from maze_consistency.model import MazeTransformer
        from maze_consistency.train import load_run, RUNS_DIR
        from maze_consistency import evaluate as E
        name = sys.argv[2] if len(sys.argv) > 2 else "tf"
        n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        maze, d = load()
        tok = Tokenizer(maze)
        params, cfg = load_run(name)
        model = MazeTransformer(cfg)
        table, per_bin, counts = E.mode_metrics(params, model, tok, maze, d)
        print("held-out next-token metrics by MODE setting:")
        for k, v in table.items():
            print(f"  {k:14s} action log-loss {v['action_nll']:.4f}  action acc {v['action_acc']:.3f}  "
                  f"dynamics acc {v['dyn_acc']:.4f}  dynamics log-loss {v['dyn_nll']:.4f}")
        print("action log-loss by true bin (counts):", list(zip(range(maze.K), counts.tolist(), [round(x, 3) for x in per_bin["R = true bin"]])))
        sweep = E.return_sweep(params, model, tok, maze, n_per=n_per)
        data_norm = float((d["returns"] / maze.R_opt(d["positions"][:, 0].astype(np.int64))).mean())
        print(f"in-maze rollouts from random starts, {n_per} per setting (return / optimal; random-walk data {data_norm:.3f}):")
        for r in sweep:
            print(f"  {r['mode']:7s} return/optimal {r['norm_return']:.3f}  reach {r['reach']:.2f}  "
                  f"P(achieved = requested) {r['hit_rate']:.2f}  excess steps when reached {r['mean_excess_steps']:.1f}")
        path = f"{RUNS_DIR}/{name}/eval.png"
        E.plot_eval(maze, d, table, per_bin, counts, sweep, path)
        print("wrote", path)
    elif cmd == "sweep":
        from maze_consistency.experiments import SweepConfig, run_sweep
        run_sweep(SweepConfig(steps=int(sys.argv[2])) if len(sys.argv) > 2 else SweepConfig())
    elif cmd == "sweep-plot":
        from maze_consistency.experiments import plot_sweep
        plot_sweep()
        print("wrote runs/sweep/curves.png")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
