"""Evaluation of a trained model.

mode_metrics   next-token metrics on held-out rollouts with the MODE slot set to NOR, the true outcome bin,
               or a wrong bin: action log-loss and accuracy (the policy), next-cell accuracy (the dynamics)
return_sweep   the model acts in the real maze with MODE = NOR or each outcome bin; per setting: discounted
               return, reach rate, and the distribution of outcome bins actually achieved
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .env import N_ACTIONS
from .tokens import Tokenizer
from .model import make_forward
from .train import N_HELDOUT


def _log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def mode_metrics(params, model, tok: Tokenizer, maze, d, n=N_HELDOUT, seed=0, chunk=100):
    fwd = make_forward(model, tok)
    idx = np.arange(len(d["length"]) - N_HELDOUT, len(d["length"]))[:n]
    L, reached, acts = d["length"][idx], d["reached"][idx], d["actions"][idx].astype(np.int64)
    cells = d["positions"][idx][:, 1:].astype(np.int64)
    valid = np.arange(maze.T)[None] < L[:, None]
    true_bins = maze.outcome_bin(L, reached)
    wrong_bins = (true_bins + np.random.default_rng(seed).integers(1, maze.K, n)) % maze.K
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], L)
    table, per_bin = {}, {}
    for name, bins in (("NOR", None), ("R = true bin", true_bins), ("R = wrong bin", wrong_bins)):
        x = tok.with_mode(body, bins)
        outs = [fwd(params, jnp.asarray(x[i:i + chunk])) for i in range(0, n, chunk)]
        pi = np.concatenate([np.asarray(o["pi_logits"][:, :maze.T]) for o in outs])
        dyn = np.concatenate([np.asarray(o["dyn_logits"]) for o in outs])
        a_nll = -np.take_along_axis(_log_softmax(pi), acts[..., None], -1)[..., 0]
        c_nll = -np.take_along_axis(_log_softmax(dyn), cells[..., None], -1)[..., 0]
        table[name] = dict(action_nll=float(a_nll[valid].mean()),
                           action_acc=float((pi.argmax(-1) == acts)[valid].mean()),
                           dyn_acc=float((dyn.argmax(-1) == cells)[valid].mean()),
                           dyn_nll=float(c_nll[valid].mean()))
        per_bin[name] = [float(a_nll[valid & (true_bins[:, None] == k)].mean()) if (true_bins == k).any() else np.nan
                         for k in range(maze.K)]
    counts = np.bincount(true_bins, minlength=maze.K)
    return table, per_bin, counts


def make_action_logits(model, tok: Tokenizer):
    types = jnp.asarray(tok.types)

    @jax.jit
    def f(params, x, i):
        out = model.apply({"params": params}, x, types[: x.shape[1]])["next"]
        return out[:, i, tok.OUT_ACT0:tok.OUT_ACT0 + N_ACTIONS]

    return f


def rollout(params, action_logits, tok: Tokenizer, maze, mode_bins, starts, rng, greedy=False, bucket=64):
    """The model acts in the real maze from `starts`. mode_bins [N]: an outcome bin per rollout, or -1 for NOR.
    Each step re-encodes the prefix, cropped to a multiple of `bucket` slots to limit recompiles."""
    mode_bins, starts = np.asarray(mode_bins), np.asarray(starts)
    N, T = len(mode_bins), maze.T
    x = tok.blank(N)
    x[:, 0] = tok.mode(np.maximum(mode_bins, 0))
    x[mode_bins < 0, 0] = tok.mode(None)
    x[:, 1] = tok.pos(starts)
    pos = starts.astype(np.int64).copy()
    length = np.full(N, T, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    for t in range(T):
        si = int(tok.sidx[t])
        Lb = min(tok.L, (si // bucket + 1) * bucket)
        lg = np.asarray(action_logits(params, jnp.asarray(x[:, :Lb]), si))
        p = np.exp(_log_softmax(lg))
        a = p.argmax(-1) if greedy else np.minimum((rng.random(N)[:, None] > p.cumsum(-1)).sum(-1), N_ACTIONS - 1)
        pos = np.where(alive, maze.next_open[pos, a], pos)
        x[alive, si + 1] = tok.act(a[alive])
        x[alive, si + 2] = tok.pos(pos[alive])
        arrived = alive & (pos == maze.goal)
        length[arrived] = t + 1
        alive &= ~arrived
        if not alive.any():
            break
    reached = ~alive
    # positions / actions in the dataset's layout, so a rollout batch can be fed straight to
    # consistency.rollout_batch or Tokenizer.encode_body.
    positions = np.zeros((N, T + 1), dtype=np.int32)
    positions[:, 0] = starts
    positions[:, 1:] = x[:, tok.sidx[1:], 1] + x[:, tok.sidx[1:], 2] * maze.W
    actions = x[:, tok.aidx, 0].astype(np.int8) - tok.ACT0
    steps = np.arange(T)[None, :] < length[:, None]
    positions[:, 1:] = np.where(steps, positions[:, 1:], maze.goal)
    return dict(starts=starts, length=length, reached=reached, returns=maze.return_of(length, reached),
                bins=maze.outcome_bin(length, reached),
                positions=positions, actions=np.where(steps, actions, 0))


def return_sweep(params, model, tok: Tokenizer, maze, n_per=64, seed=0, greedy=False):
    """MODE settings: NOR, 'best' (each start's own optimal bin), 'best far' (the same, starts >= FAR steps from
    the goal, where the data has no optimal-bin examples), and every bin. Other starts are uniform over open cells.
    norm_return = R / gamma**dist(start), so 1.0 is optimal from any start."""
    FAR = 10
    rng = np.random.default_rng(seed)
    names = ["NOR", "best", f"best far"] + [f"bin {k}" for k in range(maze.K)]
    codes = np.array([-1, -2, -3] + list(range(maze.K)))
    setting = np.repeat(np.arange(len(codes)), n_per)
    starts = rng.choice(maze.start_cells, len(setting))
    far = codes[setting] == -3
    starts[far] = rng.choice(maze.start_cells[maze.dist[maze.start_cells] >= FAR], far.sum())
    req = codes[setting]
    req = np.where(req <= -2, maze.best_bin(starts), req)
    ro = rollout(params, make_action_logits(model, tok), tok, maze, req, starts, rng, greedy)
    norm = ro["returns"] / maze.R_opt(starts)
    excess = ro["length"] - maze.dist[starts]
    rows = []
    for i, name in enumerate(names):
        sel = setting == i
        hit = sel & ro["reached"]
        rows.append(dict(mode=name, mean_return=float(ro["returns"][sel].mean()), norm_return=float(norm[sel].mean()),
                         reach=float(ro["reached"][sel].mean()),
                         hit_rate=float((ro["bins"][sel] == req[sel]).mean()) if codes[i] != -1 else float("nan"),
                         mean_excess_steps=float(excess[hit].mean()) if hit.any() else float("nan"),
                         achieved=(np.bincount(ro["bins"][sel], minlength=maze.K) / sel.sum()).tolist()))
    return rows


def plot_eval(maze, d, table, per_bin, counts, sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    names = list(table)
    ax[0, 0].bar(names, [table[n]["action_nll"] for n in names], color=["gray", "C2", "C3"])
    ax[0, 0].axhline(np.log(4), color="k", ls="--", lw=1, label="log 4 (uniform)")
    for i, n in enumerate(names):
        ax[0, 0].text(i, table[n]["action_nll"], f"acc {table[n]['action_acc']:.2f}\ndyn acc {table[n]['dyn_acc']:.3f}",
                      ha="center", va="bottom", fontsize=8)
    ax[0, 0].set_ylabel("action log-loss (nats)"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("held-out next-action log-loss by MODE setting", fontsize=10)
    ks = np.arange(maze.K)
    for n, c in (("NOR", "gray"), ("R = true bin", "C2"), ("R = wrong bin", "C3")):
        ax[0, 1].plot(ks, per_bin[n], "o-", color=c, label=n)
    ax[0, 1].axhline(np.log(4), color="k", ls="--", lw=1)
    ax[0, 1].set_xticks(ks); ax[0, 1].set_xticklabels([f"{k}\nn={c}" for k, c in enumerate(counts)], fontsize=7)
    ax[0, 1].set_xlabel("true outcome bin of the held-out rollout"); ax[0, 1].set_ylabel("action log-loss")
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("action log-loss by the rollout's true outcome bin", fontsize=10)
    modes = [r["mode"] for r in sweep]
    ax[1, 0].bar(modes, [r["norm_return"] for r in sweep], color=["gray", "C1", "C3"] + ["C0"] * maze.K)
    data_norm = float((d["returns"] / maze.R_opt(d["positions"][:, 0].astype(np.int64))).mean())
    ax[1, 0].axhline(data_norm, color="k", ls="--", lw=1, label=f"random walk (data) = {data_norm:.3f}")
    ax[1, 0].axhline(1.0, color="r", ls=":", lw=1, label="optimal = 1")
    for i, r in enumerate(sweep):
        if r["mode"] != "NOR":
            ax[1, 0].text(i, r["norm_return"], f"{r['hit_rate']:.2f}", ha="center", va="bottom", fontsize=7)
    ax[1, 0].tick_params(axis="x", rotation=60); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_ylabel("mean return / optimal return from the start")
    ax[1, 0].set_title("in-maze return by MODE (labels: P(achieved bin = requested))", fontsize=10)
    A = np.array([r["achieved"] for r in sweep])
    im = ax[1, 1].imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    ax[1, 1].set_yticks(range(len(modes))); ax[1, 1].set_yticklabels(modes, fontsize=8)
    ax[1, 1].set_xticks(ks); ax[1, 1].set_xlabel("achieved outcome bin")
    ax[1, 1].set_title("P(achieved bin | requested MODE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig
