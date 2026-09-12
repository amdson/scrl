"""Experiment sweeps: train several loss configurations and track exact-test-set metrics during training.

Nothing runs on import. Entry points:
    run_sweep(SweepConfig(...))   train every (config, seed) pair into runs/<prefix>/<config>_s<seed>/;
                                  runs whose history.json already exists are skipped
    plot_sweep(prefix)            test-metric curves over training, one panel per (metric, setting),
                                  one line per config (mean over seeds, band = min..max)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .dataset import canonical_maze
from .tokens import Tokenizer
from .testset import load_testset, stratified_rows, score
from .train import LossConfig, RUNS_DIR, train

CONFIGS = {
    "base": LossConfig(),                    # next-token loss only (NOR + R teacher forcing)
    "mc":   LossConfig(mc=True),             # + Monte Carlo value
    "td":   LossConfig(td=True),             # + TD value, identity (B)
    "mc_a4": LossConfig(mc=True, a=True),    # + Monte Carlo value + identity (A)
    "td_a4": LossConfig(td=True, a=True),    # + TD value + identity (A): the full method
}
# "a4": A runs as its own steps, LossConfig.a_updates = 4 per training step on a_batch = 16 fresh rollouts each,
# with a separate Adam at lr_a = 1e-4 (main loss: 1e-3). Earlier sweeps named mc_a / td_a put A inside the main
# loss with 16 examples per step; their runs stay on disk and load_sweep still plots them.


@dataclass(frozen=True)
class SweepConfig:
    configs: tuple = tuple(CONFIGS)
    seeds: tuple = (0,)
    steps: int = 3000
    batch: int = 32
    d_model: int = 64
    n_layers: int = 2
    eval_every: int = 250
    eval_per_setting: int = 50        # exact-test rows per setting scored at each checkpoint
    prefix: str = "sweep"


def run_name(sc: SweepConfig, config: str, seed: int) -> str:
    return f"{sc.prefix}/{config}_s{seed}"


def run_one(sc: SweepConfig, config: str, seed: int, log=print):
    tok = Tokenizer(canonical_maze())
    ts = load_testset()
    rows = stratified_rows(ts, sc.eval_per_setting, seed=0)    # the same rows for every run and checkpoint

    def eval_fn(params, fwd):
        m = score(params, fwd, tok, ts, rows)
        m.pop("per_setting")
        return m

    return train(name=run_name(sc, config, seed), steps=sc.steps, batch=sc.batch, d_model=sc.d_model,
                 n_layers=sc.n_layers, seed=seed, loss=CONFIGS[config], eval_fn=eval_fn,
                 eval_every=sc.eval_every, log=log)


def run_sweep(sc: SweepConfig = SweepConfig(), skip_existing=True, log=print):
    for config in sc.configs:
        for seed in sc.seeds:
            if skip_existing and os.path.exists(os.path.join(RUNS_DIR, run_name(sc, config, seed), "history.json")):
                log(f"[skip] {run_name(sc, config, seed)} exists")
                continue
            run_one(sc, config, seed, log=log)


def load_sweep(prefix="sweep") -> dict:
    """{config: [test history per seed]}, each history a list of {step, "<metric>/<setting>": value}."""
    root = os.path.join(RUNS_DIR, prefix)
    out = {}
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, d, "history.json")
        if os.path.exists(p):
            with open(p) as f:
                out.setdefault(d.rsplit("_s", 1)[0], []).append(json.load(f)["test"])
    return out


def plot_sweep(prefix="sweep", metrics=("act_kl", "value_kl"), settings=("NOR", "bin 10", "bin 11", "best far"),
               path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load_sweep(prefix)
    fig, ax = plt.subplots(len(metrics), len(settings), figsize=(4.2 * len(settings), 3.4 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for j, setting in enumerate(settings):
            a = ax[i, j]
            key = f"{metric}/{setting}"
            for c, (config, hists) in enumerate(runs.items()):
                steps = [m["step"] for m in hists[0]]
                ys = np.array([[m.get(key, np.nan) for m in h] for h in hists])
                a.plot(steps, ys.mean(0), color=f"C{c}", label=config)
                if len(hists) > 1:
                    a.fill_between(steps, ys.min(0), ys.max(0), color=f"C{c}", alpha=0.15)
            a.set_yscale("log")
            a.set_title(key, fontsize=9)
            a.set_xlabel("step")
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = path or os.path.join(RUNS_DIR, prefix, "curves.png")
    fig.savefig(path, dpi=110)
    return fig
