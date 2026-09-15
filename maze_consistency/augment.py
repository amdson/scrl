"""Model-generated training data: continue real prefixes with the model, asking for a higher reward.

For a sampled training rollout, pick a switch time tau, keep its prefix h_tau, and let the model continue in
R mode with MODE set above the outcome the original achieved. The continuation steps the REAL maze
(maze.next_open), not the model's dynamics head, so every spliced trajectory is a genuine environment
trajectory under a mixed behaviour policy: uniform random walk until tau, then the model.

Relabelling. R is recomputed from the spliced trajectory's own length and arrival -- the bin it ACHIEVED,
never the one it was asked for. Asking only shapes which states get visited.

Every head trains on it. The interval identity p(x | h, R) q_i(R) = p(x | h) q_j(R) is a statement about one
joint distribution. If only the R head saw the rollouts while the NOR and value heads kept the random walk,
the heads would model different joints and the consistency term would fight the data. So the model as a
whole learns the mixture: train(mixer=...) spreads rollout rows across the NOR and R halves of every batch,
and into the consistency batch.

What that does to evaluation. The DP ground truth -- the test set's act_kl and value_kl, piR*, the enrichment
ceiling -- describes the uniform random walk. Once the model learns the mixture those are references for a
different distribution: act_kl and value_kl can get WORSE while behaviour gets better, and enrichment above
100% of the random-walk ceiling is possible and good. RolloutBuffer.history is the readout that stays valid:
asked for a higher reward from a real prefix, how often does the model get one, against the exact rate a
random walk would manage from the same (tau, s_tau)?

Switch times are drawn independently of each trajectory's future: tau ~ U{0..tau_max}, keeping rows still
running at tau. Drawing tau ~ U(0, L) instead would make the switch time depend on how long the original ran.

td is incompatible: its importance weight assumes the uniform behaviour policy (train() refuses the pair).
"""
from __future__ import annotations

import numpy as np

from .dp import compute_ground_truth
from .evaluate import make_action_logits, continue_rollout
from .testset import FAR

KEYS = ("positions", "actions", "length", "reached")


def splice(params, action_logits, tok, maze, d, pool, n, rng, tau_max=None, request="above", gt=None,
           max_tries=50):
    """n spliced trajectories: a real prefix h_tau from `d` (rows drawn from `pool`), continued by the model in
    R mode, asking for a bin strictly above the one the original trajectory achieved.

    request="above": MODE uniform over the bins above the original's, up to the best still achievable from
    (tau, s_tau). request="best": always the best still achievable. Rows that had finished by tau, or from
    which no higher bin is reachable, are resampled.

    Returns the dataset layout (positions, actions, length, reached -- so the outcome bin is the one ACHIEVED)
    plus tau, src, requested, orig_bin, orig_reached, achieved. With gt (the DP) it also returns p_improve_rw
    and p_request_rw: the exact probabilities that a uniform random walk continuing from the same (tau, s_tau)
    would beat the original's bin / reach the requested one -- the per-row baseline for the model."""
    if request not in ("above", "best"):
        raise ValueError(f"request={request!r} not in above|best")
    T, K = maze.T, maze.K
    tau_max = T - 1 if tau_max is None else int(min(tau_max, T - 1))
    pool = np.asarray(pool)
    picks, have = [], 0
    for _ in range(max_tries):
        m = max(4 * (n - have), 64)
        idx = rng.choice(pool, m)
        tau = rng.integers(0, tau_max + 1, m)
        L = d["length"][idx].astype(np.int64)
        s_tau = d["positions"][idx, tau].astype(np.int64)
        togo = tau + maze.dist[s_tau]                                   # fastest possible total length
        orig = maze.outcome_bin(L, d["reached"][idx]).astype(np.int64)
        best = np.where(togo <= T, maze.success_bin(np.clip(togo, 1, T)), 0).astype(np.int64)
        ok = (L > tau) & (togo <= T) & (best > orig)
        if ok.any():
            picks.append((idx[ok], tau[ok], orig[ok], best[ok]))
            have += int(ok.sum())
        if have >= n:
            break
    if have < n:
        raise RuntimeError(f"found only {have} of {n} prefixes with a higher reward still available")
    idx, tau, orig, best = (np.concatenate(z)[:n] for z in zip(*picks))
    req = best if request == "best" else rng.integers(orig + 1, best + 1)

    ro = continue_rollout(params, action_logits, tok, maze, d["positions"][idx], d["actions"][idx], tau, req, rng)
    out = dict(positions=ro["positions"].astype(np.int16), actions=ro["actions"].astype(np.int8),
               length=ro["length"].astype(np.int16), reached=ro["reached"].astype(bool),
               tau=tau, src=idx, requested=req.astype(np.int64), orig_bin=orig,
               orig_reached=d["reached"][idx].astype(bool),
               achieved=maze.outcome_bin(ro["length"], ro["reached"]).astype(np.int64))
    if gt is not None:
        h = gt.h[tau, d["positions"][idx, tau].astype(np.int64)]          # [n, K] random walk from here on
        k = np.arange(K)[None]
        out["p_improve_rw"] = (h * (k > orig[:, None])).sum(-1)
        out["p_request_rw"] = (h * (k >= out["requested"][:, None])).sum(-1)
    return out


def summarize(maze, b) -> dict:
    """One refresh's behavioural readout. improved / got_request are the model; *_rw are what a uniform random
    walk would do from the same prefixes (exact, from the DP). far_own_best counts spliced trajectories from
    far starts that achieved their start's best bin -- the data has 0 of these in 66k far-start rollouts."""
    s0 = b["positions"][:, 0].astype(np.int64)
    far = maze.dist[s0] >= FAR
    out = dict(n=int(len(b["length"])), mean_tau=float(b["tau"].mean()),
               reach=float(b["reached"].mean()), reach_orig=float(b["orig_reached"].mean()),
               bin_requested=float(b["requested"].mean()), bin_achieved=float(b["achieved"].mean()),
               bin_orig=float(b["orig_bin"].mean()),
               improved=float((b["achieved"] > b["orig_bin"]).mean()),
               got_request=float((b["achieved"] >= b["requested"]).mean()),
               far_own_best=int((far & (b["achieved"] == maze.best_bin(s0))).sum()))
    if "p_improve_rw" in b:
        out["improved_rw"] = float(b["p_improve_rw"].mean())
        out["got_request_rw"] = float(b["p_request_rw"].mean())
    return out


class RolloutBuffer:
    """A pool of spliced trajectories, regenerated from the current params every `refresh_every` steps once
    training passes `start_after`. Duck-typed for train(mixer=...): maybe_refresh(params, step), ready,
    rows(rng, n). `model` must be the same architecture train() builds (same d_model / n_layers / n_heads).

    history holds summarize() at every refresh -- the behavioural readout that stays valid when the model is
    trained on its own rollouts (see the module docstring)."""

    def __init__(self, model, tok, maze, d, pool, size=512, refresh_every=500, start_after=1000, tau_max=100,
                 request="above", seed=0, log=print):
        self.tok, self.maze, self.d, self.pool = tok, maze, d, np.asarray(pool)
        self.size, self.refresh_every, self.start_after = size, refresh_every, start_after
        self.tau_max, self.request, self.log = tau_max, request, log
        self.action_logits = make_action_logits(model, tok)
        self.gt = compute_ground_truth(maze)
        self.rng = np.random.default_rng(10_007 + seed)                  # its own stream: data order is unchanged
        self.data, self.last, self.history = None, None, []

    @property
    def ready(self) -> bool:
        return self.data is not None

    def maybe_refresh(self, params, step) -> bool:
        if step < self.start_after or (self.last is not None and step - self.last < self.refresh_every):
            return False
        self.data = splice(params, self.action_logits, self.tok, self.maze, self.d, self.pool, self.size,
                           self.rng, self.tau_max, self.request, gt=self.gt)
        self.last = step
        s = dict(step=int(step), **summarize(self.maze, self.data))
        self.history.append(s)
        self.log(f"  rollouts@{step}: improved {s['improved']:.2f} (random walk {s['improved_rw']:.2f})  "
                 f"got request {s['got_request']:.2f} (rw {s['got_request_rw']:.2f})  "
                 f"reach {s['reach']:.2f} (orig {s['reach_orig']:.2f})  far own-best {s['far_own_best']}")
        return True

    def rows(self, rng, n) -> dict:
        i = rng.integers(0, len(self.data["length"]), n)
        return {k: self.data[k][i] for k in KEYS}
