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
from . import consistency as C

CONFIGS = {
    "base": LossConfig(),                    # next-token loss only (NOR + R teacher forcing)
    "mc":   LossConfig(mc=True),             # + Monte Carlo value
    "td":   LossConfig(td=True),             # + TD value, identity (B)
    "mc_a4": LossConfig(mc=True, a=True),    # + Monte Carlo value + identity (A)
    "td_a4": LossConfig(td=True, a=True),    # + TD value + identity (A): the full method
}
# Interval consistency (consistency_losses.md). Every variant is "mc" plus one consistency objective, so the
# data objective is identical across the whole comparison and the only difference is the added term.
#
# CONS_LAMBDA is picked per objective so each starts at a comparable gradient pull on the shared weights
# (~10% of the data loss's gradient norm, measured in consistency_demo.ipynb section 5 on a 400-step model).
# Equal loss VALUES would not mean equal influence: raw `all` and `poly_len` sit ~100x above the scaled family,
# which is exactly why they need a ~20x smaller lambda. These are a starting point, not a tuned optimum --
# sweep lambda with cons_lambda_configs() before reading much into a ranking.
CONS_LAMBDA = {"local": 0.15, "all": 0.011, "all_scaled": 0.14,
               "mixed": 0.15, "poly_len": 0.007, "multiscale": 0.15}

CONS_CONFIGS = {f"mc_{k}": LossConfig(mc=True, cons=True, cons_loss=k, w_cons=CONS_LAMBDA[k]) for k in C.ALL}
CONFIGS.update(CONS_CONFIGS)

#: the consistency comparison: the mc baseline plus every objective, all sharing one data loss
CONS_SWEEP = ("mc",) + tuple(CONS_CONFIGS)


def cons_lambda_configs(loss="all_scaled", factors=(0.1, 1.0, 10.0)):
    """Configs varying lambda_cons around the CONS_LAMBDA default for one objective, for a sensitivity sweep.
    Register them with CONFIGS.update(...) before running a sweep that names them."""
    base = CONS_LAMBDA[loss]
    return {f"mc_{loss}_x{f:g}": LossConfig(mc=True, cons=True, cons_loss=loss, w_cons=base * f) for f in factors}


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


def load_sweep(prefix="sweep", which="test") -> dict:
    """{config: [history per seed]}. which="test": exact-test metrics at each checkpoint, each entry
    {step, "<metric>/<setting>": value}. which="train": the logged loss parts, including the consistency
    term and its collapse diagnostics (cons, cond_gap, info_gain) for runs that trained one."""
    root = os.path.join(RUNS_DIR, prefix)
    out = {}
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, d, "history.json")
        if os.path.exists(p):
            with open(p) as f:
                out.setdefault(d.rsplit("_s", 1)[0], []).append(json.load(f)[which])
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


def plot_cons_diagnostics(prefix="sweep", path=None):
    """Train-side view of the consistency runs: the objective itself, then the two collapse diagnostics.

    Every consistency loss has the same degenerate global optimum -- v_t == u_t and b_t flat in t, i.e. ignore R
    -- which sends cond_gap and info_gain to 0 while the loss falls. A run whose `cons` curve drops while either
    diagnostic decays toward 0 has bought agreement by discarding the reward channel; its lambda is too high.
    Configs that trained no consistency term are drawn only in the `tf` panel, as the reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load_sweep(prefix, which="train")
    keys = [("tf", "data: teacher-forced next-token loss"), ("cons", "consistency objective"),
            ("cond_gap", "cond_gap: mean |v_t - u_t|  (-> 0 = R ignored)"),
            ("info_gain", "info_gain: log q_n(R) - log q_0(R)  (-> 0 = reward head flat)")]
    fig, ax = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8), squeeze=False)
    for j, (key, title) in enumerate(keys):
        a = ax[0, j]
        for c, (config, hists) in enumerate(sorted(runs.items())):
            steps = [m["step"] for m in hists[0]]
            ys = np.array([[m.get(key, np.nan) for m in h] for h in hists], dtype=float)
            if np.isnan(ys).all():
                continue
            a.plot(steps, np.nanmean(ys, 0), color=f"C{c}", label=config)
            if len(hists) > 1:
                a.fill_between(steps, np.nanmin(ys, 0), np.nanmax(ys, 0), color=f"C{c}", alpha=0.15)
        if key in ("tf", "cons"):
            a.set_yscale("log")
        else:
            a.axhline(0, color="k", lw=0.8, ls=":")
        a.set_title(title, fontsize=9)
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = path or os.path.join(RUNS_DIR, prefix, "cons_diagnostics.png")
    fig.savefig(path, dpi=110)
    return fig
