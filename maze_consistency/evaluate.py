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

from .dp import compute_ground_truth
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


def make_state_logits(model, tok: Tokenizer):
    """jit'd (params, x, i) -> logits [B, 5] at state slot i: the 4 actions and END. Sampling from this instead
    of the action slice lets the model end its own rollout (learned termination); no goal check is involved."""
    types = jnp.asarray(tok.types)

    @jax.jit
    def f(params, x, i):
        out = model.apply({"params": params}, x, types[: x.shape[1]])["next"]
        return jnp.concatenate([out[:, i, tok.OUT_ACT0:tok.OUT_ACT0 + N_ACTIONS], out[:, i, tok.OUT_END:tok.OUT_END + 1]], -1)

    return f


def make_cell_logits(model, tok: Tokenizer):
    """jit'd (params, x, i) -> next-cell logits at slot i (an action slot): the model's own world model."""
    types = jnp.asarray(tok.types)

    @jax.jit
    def f(params, x, i):
        out = model.apply({"params": params}, x, types[: x.shape[1]])["next"]
        return out[:, i, tok.OUT_CELL0:tok.OUT_END]

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
    return _trajectory(tok, maze, x, starts, length, ~alive)


def _trajectory(tok: Tokenizer, maze, x, starts, length, reached):
    """Read a finished rollout's token buffer back into the dataset's layout, so it can be fed straight to
    consistency.rollout_batch, Tokenizer.encode_body, or mixed into training batches."""
    N, T = len(length), maze.T
    positions = np.zeros((N, T + 1), dtype=np.int32)
    positions[:, 0] = starts
    positions[:, 1:] = x[:, tok.sidx[1:], 1] + x[:, tok.sidx[1:], 2] * maze.W
    actions = x[:, tok.aidx, 0].astype(np.int8) - tok.ACT0
    steps = np.arange(T)[None, :] < length[:, None]
    positions[:, 1:] = np.where(steps, positions[:, 1:], maze.goal)          # frozen at the goal after arrival
    return dict(starts=starts, length=length, reached=reached, returns=maze.return_of(length, reached),
                bins=maze.outcome_bin(length, reached),
                positions=positions, actions=np.where(steps, actions, 0))


def continue_rollout(params, action_logits, tok: Tokenizer, maze, positions, actions, tau, mode_bins, rng,
                     greedy=False, bucket=64, cell_logits=None, learned_end=False, max_steps=None):
    """Continue real trajectories from their prefixes h_tau.

    positions [N, T+1] / actions [N, T]: source trajectories in the dataset's layout. tau [N]: the switch time
    per row -- each source must still be running at tau (tau < its length). mode_bins [N]: MODE for the
    continuation's actions (-1 = NOR). Steps before tau are copied verbatim.

    cell_logits (make_cell_logits) makes the continuation IMAGINED: the next cell is sampled from the model's
    own dynamics head, read in NOR mode -- the unconditioned world model p(s' | h, a). Conditioning the
    dynamics on a requested high R would bias it toward lucky transitions (note section 7), and asking for high
    R selects for exactly those. No exact transitions or distances are accessed in imagined mode.
    cell_logits=None steps the real maze for EVALUATION ONLY. Training callers must provide cell_logits.

    learned_end=True: `action_logits` must come from make_state_logits (actions + END); the model ends its own
    rollout by emitting END and the goal cell is never consulted, so the result is legal as training input.
    `reached` then means "the model emitted END". Requires cell_logits (imagined dynamics).
    max_steps: stop each continuation after this many imagined steps (length = tau + max_steps if it has not
    ended); the interval identity needs no terminal, so truncated rollouts are valid consistency rows and cost
    max_steps model calls instead of up to T.

    Returns the same dict as rollout(), plus the raw token buffer "x". Use rollout_diagnostics() afterwards to
    evaluate imagined moves."""
    positions, actions = np.asarray(positions), np.asarray(actions)
    if learned_end and cell_logits is None:
        raise ValueError("learned_end requires cell_logits: termination and dynamics must both be the model's")
    tau, mode_bins = np.asarray(tau).astype(np.int64), np.asarray(mode_bins)
    N, T = len(tau), maze.T
    x = tok.with_mode(tok.encode_body(positions, actions, tau), np.maximum(mode_bins, 0))
    x[mode_bins < 0, 0] = tok.mode(None)
    pos = positions[np.arange(N), tau].astype(np.int64)
    if not learned_end:
        assert (pos != maze.goal).all(), "every source trajectory must still be running at its tau"
    length = np.full(N, T, dtype=np.int64)
    if max_steps is not None:
        length = np.minimum(length, tau + int(max_steps))
    alive = np.ones(N, dtype=bool)                   # rows before their tau are alive but not yet acting
    for t in range(int(tau.min()), int(length.max())):
        act = alive & (t >= tau) & (t < length)
        if not act.any():
            continue
        si = int(tok.sidx[t])
        Lb = min(tok.L, (si // bucket + 1) * bucket)
        lg = np.asarray(action_logits(params, jnp.asarray(x[:, :Lb]), si))
        p = np.exp(_log_softmax(lg))
        a = p.argmax(-1) if greedy else np.minimum((rng.random(N)[:, None] > p.cumsum(-1)).sum(-1), p.shape[-1] - 1)
        if learned_end:                              # column N_ACTIONS is END: the model ends its own rollout
            ended = act & (a == N_ACTIONS)
            length[ended] = t
            alive &= ~ended
            act &= ~ended
            a = np.minimum(a, N_ACTIONS - 1)
            if not act.any():
                if not alive.any():
                    break
                continue
        x[act, si + 1] = tok.act(a[act])
        if cell_logits is None:
            nxt = maze.next_open[pos, a]
        else:
            Lc = min(tok.L, ((si + 1) // bucket + 1) * bucket)
            xn = x[:, :Lc].copy()
            xn[:, 0] = tok.mode(None)                # NOR: the unconditioned world model
            pc = np.exp(_log_softmax(np.asarray(cell_logits(params, jnp.asarray(xn), si + 1)).astype(np.float64)))
            nxt = pc.argmax(-1) if greedy else np.minimum((rng.random(N)[:, None] > pc.cumsum(-1)).sum(-1),
                                                          maze.n_cells - 1)
        pos = np.where(act, nxt, pos)
        x[act, si + 2] = tok.pos(pos[act])
        if not learned_end:
            arrived = act & (pos == maze.goal)
            length[arrived] = t + 1
            alive &= ~arrived
        if not alive.any():
            break
    out = _trajectory(tok, maze, x, positions[:, 0], length, ~alive)
    out["x"] = x
    return out


def rollout_diagnostics(maze, b, gt=None):
    """Evaluate already-generated splices using the real maze. Never mutates or filters training rows.

    Exact transitions, optimal distances, and DP baselines belong here, outside generation. Imagined
    rewards and these diagnostics do not replace evaluating the policy by acting in the real maze.
    """
    from .testset import FAR

    pos, actions = b["positions"].astype(np.int64), b["actions"].astype(np.int64)
    t = np.arange(maze.T)[None]
    tau = b["tau"]
    active = (t >= tau[:, None]) & (t < b["length"][:, None])
    n_bad = ((pos[:, 1:] != maze.next_open[pos[:, :-1], actions]) & active).sum(-1)
    s0 = pos[:, 0]
    far_best = (maze.dist[s0] >= FAR) & (b["achieved"] == maze.best_bin(s0))
    gt = compute_ground_truth(maze) if gt is None else gt
    h = gt.h[tau, pos[np.arange(len(tau)), tau]]
    k = np.arange(maze.K)[None]
    return dict(n_bad=n_bad, far_own_best=int(far_best.sum()),
                p_improve_rw=(h * (k > b["orig_bin"][:, None])).sum(-1),
                p_request_rw=(h * (k >= b["requested"][:, None])).sum(-1))


def imagined_rollout_eval(maze, ro, tau):
    """EVALUATION ONLY. Check a batch of imagined rollouts (continue_rollout output, dataset layout) against the
    real maze: the fraction of imagined steps whose next cell is not the real transition, the fraction of
    rollouts with at least one such move, and where the model's END emissions landed. Never feeds training."""
    pos, actions = ro["positions"].astype(np.int64), ro["actions"].astype(np.int64)
    tau, length = np.asarray(tau).astype(np.int64), ro["length"].astype(np.int64)
    t = np.arange(maze.T)[None]
    active = (t >= tau[:, None]) & (t < length[:, None])
    cur, pred = pos[:, :-1], pos[:, 1:]
    bad = (pred != maze.next_open[cur, actions]) & active
    n_steps = max(int(active.sum()), 1)
    # kind of error: the predicted cell is a wall; an open neighbour (or the cell itself) that the action does
    # not lead to; or an open cell that is not adjacent at all
    wall = bad & (maze.dist[pred] < 0)
    adjacent = (pred == cur) | (pred[..., None] == maze.next_open[cur]).any(-1)
    wrong_dir = bad & ~wall & adjacent
    teleport = bad & ~wall & ~adjacent
    ended = ro["reached"].astype(bool)
    at_goal = pos[np.arange(len(length)), length] == maze.goal
    return dict(invalid_step_frac=float(bad.sum() / n_steps),
                invalid_row_frac=float((bad.sum(1) > 0).mean()),
                wrong_dir_frac=float(wrong_dir.sum() / n_steps),
                wall_frac=float(wall.sum() / n_steps),
                teleport_frac=float(teleport.sum() / n_steps),
                end_at_goal=float(at_goal[ended].mean()) if ended.any() else np.nan,   # END emitted at the goal
                reached_goal=float((at_goal & ended).mean()),                          # ended, and really there
                mean_imagined_steps=float(active.sum(1).mean()))


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


def conditioning_response(params, model, tok: Tokenizer, maze, gt, cells=None, chunk=512):
    """Does the MODE token actually move the policy, and by how much? One forward pass per (cell, MODE),
    no rollouts -- so it is free of the compounding that makes return_sweep unreadable at T = 200.

    At the prefix (MODE, s) the model's action distribution is directly comparable to the exact
    piR_star[0, s, k], because that prefix carries no history for the two to disagree about. Returns per bin
    k = 0..K-1, over the cells where bin k is feasible (h[0, s, k] > 0):

        p_closer        [K+1, n]  P(model picks an action reducing distance to the goal); row 0 is NOR
        p_closer_exact  [K, n]    the same under piR*   (the NOR truth is 1/4 by construction)
        kl              [K, n]    KL(piR* || model)
        kl_uniform      [K, n]    KL(piR* || uniform) -- what a MODE-IGNORING model scores. This is the
                                  ceiling that makes `kl` readable: most of a raw act_kl is the irreducible
                                  entropy of piR*, so the fraction below is the honest measure.
        captured        [K]       1 - mean(kl) / mean(kl_uniform), the share of the conditioning signal learned
        feasible        [K, n]    h[0, s, k] > 0

    A flat p_closer across MODE settings means the model ignores the conditioning entirely, whatever its
    act_kl looks like."""
    cells = maze.start_cells if cells is None else np.asarray(cells)
    n, K = len(cells), maze.K
    fwd = make_forward(model, tok)
    closer = np.zeros((n, N_ACTIONS), bool)
    for a in range(N_ACTIONS):
        closer[:, a] = maze.dist[maze.next_open[cells, a]] < maze.dist[cells]

    def policy(mode):
        x = tok.blank(n)
        x[:, 0] = tok.mode(None if mode is None else np.full(n, mode))
        x[:, 1] = tok.pos(cells)
        lg = np.concatenate([np.asarray(fwd(params, jnp.asarray(x[i:i + chunk]))["pi_logits"][:, 0])
                             for i in range(0, n, chunk)])
        return np.exp(_log_softmax(lg.astype(np.float64)))

    from .dp import truth_for
    gt = truth_for(gt, tok.cond)
    q = np.stack([policy(m) for m in [None] + list(range(K))])            # [K+1, n, 4]
    p = np.transpose(gt.piR_star[0, cells], (1, 0, 2))                    # [K, n, 4]
    feasible = gt.h[0, cells].T > 0                                       # [K, n]
    p = np.where(feasible[..., None], np.nan_to_num(p), 0.25)

    def kl(qq):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p * (np.log(p) - np.log(np.maximum(qq, 1e-30))), 0.0).sum(-1)

    k_model, k_unif = kl(q[1:]), kl(np.full_like(p, 0.25))
    m = feasible & np.isfinite(k_model)
    captured = np.array([1 - k_model[k][m[k]].mean() / k_unif[k][m[k]].mean()
                         if m[k].any() and k_unif[k][m[k]].mean() > 0 else np.nan for k in range(K)])
    return dict(cells=cells, p_closer=(q * closer[None]).sum(-1),
                p_closer_exact=(p * closer[None]).sum(-1),
                kl=k_model, kl_uniform=k_unif, captured=captured, feasible=m,
                dist=maze.dist[cells])


def prefix_response(params, model, tok: Tokenizer, maze, gt, d, idx,
                    depth_edges=(0, 1, 4, 16, 64, 201), chunk=128):
    """Conditioning response at EVERY prefix of a set of trajectories, not just the t = 0 start.

    conditioning_response probes one prefix per cell: (MODE, s). This sweeps MODE over real prefixes drawn
    from data, at every depth, which is where the model actually operates and what the test set's act_kl
    averages over. Causal attention makes it cheap -- one forward pass per (trajectory, MODE) reads out the
    conditioned policy at all T+1 state slots at once, so the whole thing is K+1 passes.

    The exact target at prefix h_t is piR_star[t, s_t, k]: the chain is Markov in (t, s), so conditioning on
    the full history does not change it, and the model's readout is directly comparable.

    Feasibility matters much more here than at t = 0. Deep into a trajectory most bins are already ruled out
    (h[t, s_t, k] = 0, piR* undefined), so the mask is depth-dependent and `n_feasible` is reported: a bin
    that no longer has support is not a failure to condition, it is an impossible request.

    Returns per depth bucket [t_lo, t_hi):
        captured    [n_buckets, K]  1 - KL(piR*||model) / KL(piR*||uniform), the share of signal used
        p_closer    [n_buckets, K]  model P(step toward goal)
        p_exact     [n_buckets, K]  the same under piR*
        n_feasible  [n_buckets, K]  prefixes contributing
    """
    idx = np.asarray(idx)
    N, T, K = len(idx), maze.T, maze.K
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], d["length"][idx])
    L = d["length"][idx].astype(np.int64)
    pos = d["positions"][idx].astype(np.int64)[:, :T]                       # [N, T] cell at each state slot
    fwd = make_forward(model, tok)
    ts = np.arange(T)[None, :]
    valid = ts < L[:, None]                                                 # slots that actually take an action

    closer = np.zeros((N, T, N_ACTIONS), bool)
    for a in range(N_ACTIONS):
        closer[..., a] = maze.dist[maze.next_open[pos, a]] < maze.dist[pos]

    q = []
    for k in range(K):
        x = tok.with_mode(body, np.full(N, k))
        pi = np.concatenate([np.asarray(fwd(params, jnp.asarray(x[i:i + chunk]))["pi_logits"][:, :T])
                             for i in range(0, N, chunk)])
        q.append(np.exp(_log_softmax(pi.astype(np.float64))))
    q = np.stack(q)                                                         # [K, N, T, 4]

    from .dp import truth_for
    gt = truth_for(gt, tok.cond)
    p = np.transpose(gt.piR_star[ts, pos], (2, 0, 1, 3))                    # [K, N, T, 4] exact
    feas = np.transpose(gt.h[ts, pos] > 0, (2, 0, 1)) & valid[None]         # [K, N, T]
    p = np.where(feas[..., None], np.nan_to_num(p), 0.25)

    def kl(qq):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p * (np.log(p) - np.log(np.maximum(qq, 1e-30))), 0.0).sum(-1)

    k_model, k_unif = kl(q), kl(np.full_like(p, 0.25))
    ok = feas & np.isfinite(k_model) & np.isfinite(k_unif)
    pc_m, pc_e = (q * closer[None]).sum(-1), (p * closer[None]).sum(-1)

    edges = list(depth_edges)
    buckets = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    shape = (len(buckets), K)
    cap, pm, pe, nf = (np.full(shape, np.nan) for _ in range(4))
    for bi, (lo, hi) in enumerate(buckets):
        in_d = (ts >= lo) & (ts < hi)
        for k in range(K):
            m = ok[k] & in_d
            nf[bi, k] = m.sum()
            if not m.any():
                continue
            pm[bi, k], pe[bi, k] = pc_m[k][m].mean(), pc_e[k][m].mean()
            denom = k_unif[k][m].mean()
            cap[bi, k] = 1 - k_model[k][m].mean() / denom if denom > 0 else np.nan
    return dict(buckets=buckets, captured=cap, p_closer=pm, p_exact=pe, n_feasible=nf)


def make_enrichment_eval(tok: Tokenizer, maze, d, n=128, seed=0, chunk=64):
    """Factory for an eval_fn-compatible probe: does conditioning on a higher reward enrich goalward actions?

    Paired per prefix. At every step of every held-out trajectory the same model is asked twice --
        fail  MODE = bin 0
        best  MODE = success_bin(t + dist(s_t)), the best outcome still achievable from here
    -- and scored on the probability mass it puts on actions that reduce distance to the goal. Both columns
    come from the identical prefix, so the difference is conditioning alone: no feasibility or composition
    artefact. Prefixes from which the goal is already unreachable are dropped.

    Returns {"enrich/points", "enrich/frac_of_exact", "goalward/fail", "goalward/best"} plus
    "goalward_bin/<k>": the goalward mass when asking each bin k, on the same prefixes. The "/" split keeps
    them plottable by the notebooks' plot_curves(metrics=("enrich",), settings=("points", ...)).

    The exact conditional's enrichment is computed once as the ceiling -- on the canonical maze it is about
    +40 points, so frac_of_exact is a share of a large, real effect rather than of an estimated one.

    Costs K forward passes of `n` sequences per call, so keep n modest if eval_every is small."""
    n = (n // chunk) * chunk or chunk                  # keep every batch the same shape, so fwd compiles once
    n_train = len(d["length"]) - N_HELDOUT
    idx = np.random.default_rng(seed).choice(np.arange(n_train, n_train + N_HELDOUT), n, replace=False)
    T, K = maze.T, maze.K
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], d["length"][idx])
    x_all = np.stack([tok.with_mode(body, np.full(n, k)) for k in range(K)])      # [K, n, L, 3]
    pos = d["positions"][idx].astype(np.int64)[:, :T]
    ts = np.arange(T)[None, :]
    togo = ts + maze.dist[pos]
    use = (ts < d["length"][idx].astype(np.int64)[:, None]) & (togo <= T)
    k_best = np.clip(maze.success_bin(togo), 1, K - 1)
    closer = np.stack([maze.dist[maze.next_open[pos, a]] < maze.dist[pos] for a in range(N_ACTIONS)], -1)
    nn = np.arange(n)[:, None]

    # Under cond="threshold" the "fail" reference is NOR (token 0 is the sure event, which the training rows never
    # carry) and the exact conditionals are the event ground truth.
    from .dp import truth_for
    threshold = tok.cond == "threshold"
    x_nor = tok.with_mode(body, None)

    def score_q(q, q_fail=None):                       # q [K, n, T, 4] -> (fail, best, enrichment) in %
        g = (q * closer[None]).sum(-1)
        gf = g[0] if q_fail is None else (q_fail * closer).sum(-1)
        fail, best = gf[use], g[k_best, nn, ts][use]
        return 100 * fail.mean(), 100 * best.mean(), 100 * (best - fail).mean()

    gt = truth_for(compute_ground_truth(maze), tok.cond)
    p_exact = np.nan_to_num(np.transpose(gt.piR_star[ts, pos], (2, 0, 1, 3)), nan=0.25)
    ceiling = score_q(p_exact, np.full(p_exact.shape[1:], 0.25) if threshold else None)[2]

    def fn(params, fwd):
        q = []
        for k in range(K):
            lg = np.concatenate([np.asarray(fwd(params, jnp.asarray(x_all[k, i:i + chunk]))["pi_logits"][:, :T])
                                 for i in range(0, n, chunk)]).astype(np.float64)
            q.append(np.exp(_log_softmax(lg)))
        q = np.stack(q)
        q_fail = None
        if threshold:
            lg = np.concatenate([np.asarray(fwd(params, jnp.asarray(x_nor[i:i + chunk]))["pi_logits"][:, :T])
                                 for i in range(0, n, chunk)]).astype(np.float64)
            q_fail = np.exp(_log_softmax(lg))
        fail, best, delta = score_q(q, q_fail)
        out = {"enrich/points": delta, "enrich/frac_of_exact": delta / ceiling if ceiling else np.nan,
               "goalward/fail": fail, "goalward/best": best}
        g = (q * closer[None]).sum(-1)                 # [K, n, T] goalward mass when asking each bin
        for k in range(K):                             # same prefixes for every bin, so these are paired too
            out[f"goalward_bin/{k:02d}"] = 100 * g[k][use].mean()
        return out

    fn.ceiling = ceiling
    return fn
