"""E1–E5 runners and their plots. Results land in results/<E>/<run>/metrics.json."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from .env import get_maze
from .dp import compute_ground_truth
from .train import TrainConfig, train, load_run
from .fqi import tabular_fqi, biased_data
from .eval import dp_reference_rollouts
from .env import random_walk_episodes

PROFILES = {
    # laptop CPU: ~50 ms/step at these sizes
    "cpu": dict(d_model=64, n_layers=2, n_heads=4, batch=64, steps=2000, eval_every=250, eval_n=2000,
                rollout_n=512, seeds=2, Ns=(2000, 10000, 50000), e4_steps=3000, e4_seeds=3),
    # the plan's sizes
    "gpu": dict(d_model=128, n_layers=4, n_heads=4, batch=256, steps=20000, eval_every=1000, eval_n=4000,
                rollout_n=1024, seeds=3, Ns=(2000, 10000, 50000), e4_steps=20000, e4_seeds=5),
    # pipeline check
    "quick": dict(d_model=32, n_layers=1, n_heads=2, batch=32, steps=100, eval_every=50, eval_n=300,
                  rollout_n=64, seeds=1, Ns=(2000,), e4_steps=100, e4_seeds=1),
}


def _base(profile, exp, name, **kw) -> TrainConfig:
    p = PROFILES[profile]
    cfg = TrainConfig(exp=exp, name=name, d_model=p["d_model"], n_layers=p["n_layers"], n_heads=p["n_heads"],
                      batch=p["batch"], steps=p["steps"], eval_every=p["eval_every"], eval_n=p["eval_n"],
                      rollout_n=p["rollout_n"])
    return replace(cfg, **kw)


def _run(cfg: TrainConfig, force=False, log=print):
    path = os.path.join(cfg.run_dir, "metrics.json")
    if os.path.exists(path) and not force:
        log(f"[skip] {cfg.run_dir} exists")
        return load_run(cfg.run_dir)[0]
    return train(cfg, log=log)


def _a_warmup(cfg: TrainConfig) -> dict:
    return dict(a_warmup=cfg.steps // 5)


# ---------------------------------------------------------------------------
def run_e1(profile="cpu", force=False, log=print):
    """H1: value error MC vs TD across data sizes."""
    p = PROFILES[profile]
    out = []
    for N in p["Ns"]:
        for seed in range(p["seeds"]):
            for loss in ("MC", "TD"):
                cfg = _base(profile, "E1", f"{loss}_N{N}_s{seed}", loss=loss, N=N, seed=seed, data_seed=seed)
                out.append(_run(cfg, force, log))
    return out


def run_e2(profile="cpu", force=False, log=print, N=10000):
    """H2: does (A) recover pi_R at R=-d* (never observed in data)?"""
    p = PROFILES[profile]
    out = []
    for seed in range(p["seeds"]):
        for loss in ("MC", "TD", "TDA"):
            cfg = _base(profile, "E2", f"{loss}_N{N}_s{seed}", loss=loss, N=N, seed=seed, data_seed=seed)
            if loss == "TDA":
                cfg = replace(cfg, loss_kw=_a_warmup(cfg))
            out.append(_run(cfg, force, log))
    return out


def run_e2b(profile="cpu", force=False, log=print, N=10000):
    """H2 follow-up: why on-sequence TD+A cannot reach an unobserved return bin, and what does.
    rtg   = value bins over return-to-go (terminal fact shared across t)
    child = explicit 4-child (B)/(A) with analytic terminal (plan §4)
    rare  = keep the ~h0*N optimal episodes in the data (the plan's original, non-strict setting)"""
    p = PROFILES[profile]
    variants = {
        "TDA_rtg": dict(loss="TDA", rtg=True),
        "TDA_child": dict(loss="TDA", children=True),
        "TDA_rtg_child": dict(loss="TDA", rtg=True, children=True),
        "MC_rtg": dict(loss="MC", rtg=True),
        "TDA_rare": dict(loss="TDA", drop_optimal=False),
        "MC_rare": dict(loss="MC", drop_optimal=False),
    }
    out = []
    for seed in range(min(p["seeds"], 2)):
        for name, kw in variants.items():
            cfg = _base(profile, "E2b", f"{name}_N{N}_s{seed}", N=N, seed=seed, data_seed=seed, **kw)
            cfg = replace(cfg, loss_kw=_a_warmup(cfg))
            out.append(_run(cfg, force, log))
    return out


def plot_e2b(save="results/E2b/E2b.png"):
    import matplotlib.pyplot as plt
    from .viz import plot_vs_step, plot_per_t
    runs = _load_exp("E2b")
    for k, v in _load_exp("E2").items():          # include the strict on-sequence baselines
        if k.startswith("TDA") or k.startswith("MC"):
            runs["E2/" + k] = v
    g = _group(runs, lambda n, r: n.split("_N")[0])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_vs_step(ax[0], g, "opt_err", title="opt_err = KL(pi_R* || pi_R) at R=-d*", logy=True)
    ax[0].axhline(0.1, color="k", ls="--", lw=1)
    plot_per_t(ax[1], g, "logp_rstar_err", title="|log V_t(R=-d*) - log h_t(R=-d*)| (final)", logy=True)
    names = list(g)
    vals = [[r["evals"][-1]["rollouts"]["piR_sample"]["optimal_rate"] for r in g[n]] for n in names]
    ax[2].bar(range(len(names)), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals])
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax[2].set_title("P(optimal episode) sampling pi_R(R=-d*)", fontsize=9); ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("E2b: reaching an unobserved return bin — rtg bins / explicit children / rare-but-observed", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig


def run_e3(profile="cpu", force=False, log=print, N=10000):
    """H3: posterior tilt vs EV tilt on the locked-door maze."""
    p = PROFILES[profile]
    maze = get_maze("door")
    gt = compute_ground_truth(maze)
    os.makedirs("results/E3", exist_ok=True)
    ref = dp_reference_rollouts(maze, gt, N=4000)
    with open("results/E3/dp_reference.json", "w") as f:
        json.dump(ref, f, indent=1)
    out = []
    for seed in range(p["seeds"]):
        cfg = _base(profile, "E3", f"TDA_door_N{N}_s{seed}", loss="TDA", maze="door", N=N, seed=seed, data_seed=seed)
        cfg = replace(cfg, loss_kw=_a_warmup(cfg))
        out.append(_run(cfg, force, log))
    return out, ref


def run_e4(profile="cpu", force=False, log=print, N=10000, drop_actions=0.0):
    """H4: stability of TD+A vs neural FQI (and tabular FQI) on the same data."""
    p = PROFILES[profile]
    out = []
    tag = f"N{N}" + (f"_drop{drop_actions}" if drop_actions else "")
    for seed in range(p["e4_seeds"]):
        for kind in ("TDA", "FQI"):
            cfg = _base(profile, "E4", f"{kind}_{tag}_s{seed}", N=N, seed=seed, data_seed=seed, steps=p["e4_steps"],
                        drop_actions=drop_actions, rollouts_every_eval=False)
            cfg = replace(cfg, loss="TDA", loss_kw=_a_warmup(cfg)) if kind == "TDA" else replace(cfg, fqi=True)
            out.append(_run(cfg, force, log))
        # tabular FQI on exactly the same data
        path = f"results/E4/tabFQI_{tag}_s{seed}.json"
        if force or not os.path.exists(path):
            maze = get_maze("default")
            gt = compute_ground_truth(maze)
            data = random_walk_episodes(maze, N, seed, drop_optimal=True)
            if drop_actions:
                data = biased_data(data, drop_actions, seed)
            r = tabular_fqi(maze, gt, data, log=log)
            with open(path, "w") as f:
                json.dump(dict(final={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r["final"].items()},
                               history=[{k: (float(np.nanmean(v)) if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in r["history"]],
                               greedy_rollout=r["greedy_rollout"]), f)
    return out


def run_e5(profile="cpu", force=False, log=print, N=10000):
    """H5: loop closures, one at a time, on top of TD+A."""
    p = PROFILES[profile]
    out = []
    closures = {"base": {}, "c1_rollout": dict(closure_rollout=True), "c2_distill": dict(closure_distill=True),
                "c3_softq": dict(closure_softq=True)}
    for seed in range(min(p["seeds"], 2)):
        for name, kw in closures.items():
            cfg = _base(profile, "E5", f"{name}_N{N}_s{seed}", loss="TDA", N=N, seed=seed, data_seed=seed,
                        steps=p["e4_steps"], rollouts_every_eval=False, **kw)
            cfg = replace(cfg, loss_kw=_a_warmup(cfg), rollout_every=max(50, cfg.steps // 30),
                          distill_every=max(100, cfg.steps // 6))
            out.append(_run(cfg, force, log))
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _load_exp(exp):
    root = os.path.join("results", exp)
    runs = {}
    if not os.path.isdir(root):
        return runs
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "metrics.json")
        if os.path.exists(p):
            runs[d] = load_run(os.path.join(root, d))[0]
    return runs


def _group(runs, key_fn):
    g = {}
    for name, r in runs.items():
        g.setdefault(key_fn(name, r), []).append(r)
    return g


def plot_e1(save="results/E1/E1.png"):
    import matplotlib.pyplot as plt
    from .viz import plot_per_t, plot_vs_step
    runs = _load_exp("E1")
    Ns = sorted({r["data"]["N"] for r in runs.values()})
    Ns_cfg = sorted({r["config"]["N"] for r in runs.values()})
    fig, ax = plt.subplots(2, len(Ns_cfg), figsize=(4.5 * len(Ns_cfg), 7.5), squeeze=False)
    for j, N in enumerate(Ns_cfg):
        sub = {k: v for k, v in runs.items() if v["config"]["N"] == N}
        g = _group(sub, lambda n, r: r["config"]["loss"])
        plot_per_t(ax[0, j], g, "value_err", title=f"N={N}: value_err[t] (final)", logy=True)
        plot_per_t(ax[1, j], g, "logp_rstar_err", title=f"N={N}: |log V_t(R*) - log h_t(R*)|", logy=True)
    fig.suptitle("E1 (H1): MC vs TD value error per t", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig


def plot_e2(save="results/E2/E2.png"):
    import matplotlib.pyplot as plt
    from .viz import plot_vs_step
    runs = _load_exp("E2")
    g = _group(runs, lambda n, r: r["config"]["loss"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    plot_vs_step(ax[0], g, "opt_err", title="opt_err = KL(pi_R* || pi_R) at R=-d* on shortest paths", logy=True)
    ax[0].axhline(0.1, color="k", ls="--", lw=1, label="kill line 0.1")
    ax[0].legend(fontsize=7)
    names = list(g)
    for key, a, ttl in (("optimal_rate", ax[1], "P(episode is optimal) under pi_R(R=-d*) sampling"),
                        ("solve_rate", ax[2], "solve rate under pi_R(R=-d*) sampling")):
        vals = [[r["evals"][-1]["rollouts"]["piR_sample"][key] for r in g[n]] for n in names]
        a.bar(names, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], color=[{"MC": "#d62728", "TD": "#1f77b4", "TDA": "#2ca02c"}.get(n, "gray") for n in names])
        a.set_title(ttl, fontsize=9)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("E2 (H2): recovering the optimal policy from an unobserved return", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig


def plot_e3(save="results/E3/E3.png"):
    import matplotlib.pyplot as plt
    runs = _load_exp("E3")
    with open("results/E3/dp_reference.json") as f:
        ref = json.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = list(ref)
    ax[0].bar(names, [ref[n]["door_choice"] for n in names], color="gray")
    ax[0].set_title("DP reference: P(route via door) with perfect values", fontsize=9)
    ax[0].tick_params(axis="x", rotation=45)
    if runs:
        rr = [r["evals"][-1]["rollouts"] for r in runs.values()]
        names = list(rr[0])
        vals = np.array([[x[n]["door_choice"] for n in names] for x in rr])
        ax[1].bar(names, vals.mean(0), yerr=vals.std(0), color="#2ca02c")
        ax[1].set_title("model (TD+A) decodes: P(route via door)", fontsize=9)
        ax[1].tick_params(axis="x", rotation=45)
    for a in ax:
        a.set_ylim(0, 1.05)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("E3 (H3): posterior tilt gambles on the door; EV tilt", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig


def plot_e4(save="results/E4/E4.png"):
    import matplotlib.pyplot as plt
    from .viz import plot_vs_step, plot_per_t
    runs = _load_exp("E4")
    tda = {k: v for k, v in runs.items() if k.startswith("TDA")}
    fqi = {k: v for k, v in runs.items() if k.startswith("FQI")}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    if tda:
        g = _group(tda, lambda n, r: "TDA")
        plot_vs_step(ax[0, 0], g, "max_value_err", title="TD+A: max_t value_err vs step", logy=True)
        plot_vs_step(ax[0, 1], g, "mean_consistency", title="TD+A: mean (A) residual vs step", logy=True)
        for r in list(tda.values())[:1]:
            for i, e in enumerate(r["evals"]):
                ax[0, 2].plot(np.array(e["value_err"], dtype=float), color=plt.cm.viridis(i / max(len(r["evals"]) - 1, 1)), lw=1)
        ax[0, 2].set_title("TD+A: value_err[t] at checkpoints (seed 0; dark=early)", fontsize=9)
        ax[0, 2].set_yscale("log"); ax[0, 2].grid(alpha=0.3)
    if fqi:
        g = _group(fqi, lambda n, r: "FQI")
        plot_vs_step(ax[1, 0], g, "q_err", title="neural FQI: max_t |max_a Q - V_opt| vs step", logy=True)
        plot_vs_step(ax[1, 1], g, "overest", title="neural FQI: max_t overestimation vs step")
        for r in list(fqi.values())[:1]:
            for i, e in enumerate(r["evals"]):
                ax[1, 2].plot(np.array(e["q_err"], dtype=float), color=plt.cm.viridis(i / max(len(r["evals"]) - 1, 1)), lw=1)
        ax[1, 2].set_title("neural FQI: q_err[t] at checkpoints (seed 0)", fontsize=9)
        ax[1, 2].set_yscale("log"); ax[1, 2].grid(alpha=0.3)
    tabs = [f for f in os.listdir("results/E4") if f.startswith("tabFQI")] if os.path.isdir("results/E4") else []
    for f in tabs:
        with open(os.path.join("results/E4", f)) as fh:
            t = json.load(fh)
        ax[1, 2].plot(np.array(t["final"]["q_err"], dtype=float), "k--", lw=1, label="tabular FQI")
    if tabs:
        ax[1, 2].legend(fontsize=7)
    fig.suptitle("E4 (H4): stability of TD+A vs FQI", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig


def plot_e5(save="results/E5/E5.png"):
    import matplotlib.pyplot as plt
    from .viz import plot_vs_step, plot_per_t
    runs = _load_exp("E5")
    g = _group(runs, lambda n, r: n.split("_N")[0])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    plot_vs_step(ax[0], g, "max_value_err", title="max_t value_err vs step", logy=True)
    plot_per_t(ax[1], g, "value_err", title="value_err[t] (final)", logy=True)
    plot_vs_step(ax[2], g, "opt_err", title="opt_err vs step", logy=True)
    fig.suptitle("E5 (H5): loop closures on top of TD+A", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    return fig
