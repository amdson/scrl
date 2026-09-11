#!/usr/bin/env python
"""CLI.  Examples:
  python run.py truth                      # DP ground-truth overview figures for both mazes
  python run.py E1 --profile cpu           # run experiment (skips finished runs)
  python run.py plot E1                    # figure for an experiment
  python run.py report results/E2/TDA_N10000_s0   # per-run report figure
  python run.py single --loss TDA --N 10000 --steps 2000 --name mytest
"""
import argparse
import sys

from maze_consistency import experiments as X
from maze_consistency.train import TrainConfig, train
from maze_consistency.viz import run_report, plot_ground_truth
from maze_consistency.env import get_maze
from maze_consistency.dp import compute_ground_truth, check_identities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--loss", default="TDA")
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maze", default="default")
    ap.add_argument("--name", default="single")
    ap.add_argument("--drop_actions", type=float, default=0.0)
    ap.add_argument("--rtg", action="store_true")
    ap.add_argument("--children", action="store_true")
    a = ap.parse_args()
    if a.cmd == "truth":
        import os
        os.makedirs("results/truth", exist_ok=True)
        for name in ("default", "door"):
            m = get_maze(name)
            gt = compute_ground_truth(m)
            print(m, check_identities(gt))
            print(m.ascii())
            plot_ground_truth(m, gt, f"results/truth/{name}.png")
        return
    if a.cmd in ("E1", "E2", "E2b", "E3", "E4", "E5"):
        fn = getattr(X, f"run_{a.cmd.lower()}")
        kw = dict(profile=a.profile, force=a.force)
        if a.cmd == "E4" and a.drop_actions:
            kw["drop_actions"] = a.drop_actions
        fn(**kw)
        getattr(X, f"plot_{a.cmd.lower()}")()
        return
    if a.cmd == "plot":
        getattr(X, f"plot_{a.arg.lower()}")()
        print(f"wrote results/{a.arg}/{a.arg}.png")
        return
    if a.cmd == "report":
        run_report(a.arg)
        print(f"wrote {a.arg}/report.png")
        return
    if a.cmd == "single":
        p = X.PROFILES[a.profile]
        cfg = TrainConfig(exp="single", name=a.name, loss=a.loss, N=a.N, steps=a.steps, seed=a.seed, maze=a.maze,
                          d_model=p["d_model"], n_layers=p["n_layers"], batch=p["batch"], eval_every=max(a.steps // 8, 1),
                          loss_kw=dict(a_warmup=a.steps // 5), drop_actions=a.drop_actions, rtg=a.rtg, children=a.children)
        train(cfg)
        run_report(cfg.run_dir)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
