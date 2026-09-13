"""Interval consistency losses (the note in consistency_losses.md), in JAX.

Two teacher-forced passes over the same rollout -- one with MODE = NOR, one with MODE = the rollout's own
outcome bin R -- give, at every env-step boundary t:

    u_t = log p(a_t, s_{t+1} | h_t)       NOR pass, action slot + dynamics slot, full flat next-token softmax
    v_t = log p(a_t, s_{t+1} | h_t, R)    R pass, the same two slots
    b_t = log q_t(R) = log p(R | h_t)     NOR pass value head at state slot t,  t = 0..n

A block is one whole env step (note section 7, "choose boundaries deliberately"): the value head is only read
at state slots, so an interval endpoint always sits between a state and the action that follows it. The END
token predicted at the last state slot is left out of every block -- it belongs to the data loss, and in this
maze it is determined by s_n anyway. u, v use the full flat softmax, not the action-only or cell-only slice:
renormalizing per slot type adds a log-partition term to delta that does not telescope.

    delta_t = v_t - u_t + b_{t-1} - b_t      one-step residual, t = 1..n
    c_0 = 0, c_k = sum_{t<=k} delta_t        so Delta_{i,j} = c_j - c_i for all n(n+1)/2 intervals

Losses, all O(n) or O(n log n):
    local           mean_t delta_t^2
    all_intervals   mean over i<j of (c_j - c_i)^2, as 2(n+1)/n * Var(c_0..c_n)     [note section 3]
    poly            the same under a weight polynomial in interval length           [extension, below]
    multiscale      equal weight per power-of-two length, each divided by length    [note section 3]

`poly` extends the variance shortcut past uniform weighting: any w(l) = sum_d coefs[d] * l^d keeps an O(dn)
form, because sum_{i<j} j^r i^s (c_j - c_i)^2 splits into prefix sums of c_k^2, c_k and k^m. coefs=[1.0]
reproduces all_intervals exactly; coefs=[0.0, 1.0] weights an interval by its length. The note's 1/l
equalizing weight is not polynomial -- that case needs multiscale (or an FFT autocorrelation, not implemented).

all_intervals hides a strongly centre-peaked weighting of the individual steps: it is delta^T A delta with
A[s,t] = min(s,t) * (n+1-max(s,t)), so the first and last steps carry O(n) weight and the middle O(n^2/4).
weight_profile() returns that diagonal; the brute-force references at the bottom are for tests and for the
by-length diagnostics.
"""
from __future__ import annotations

from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .env import N_ACTIONS as N_ACTIONS_


# ---- residuals ---------------------------------------------------------------------------------

def residuals(u, v, b, lengths):
    """u, v [B, N]; b [B, N+1]; lengths [B] (env steps per rollout, >= 1). All arrays float.

    Returns delta [B, N], c [B, N+1], and the masks/lengths the losses need. Entries past a rollout's own
    length are zeroed before any arithmetic, so c is flat (not repeating) outside the valid range."""
    u, v, b = (jnp.asarray(z, jnp.float32) for z in (u, v, b))
    B, N = u.shape
    lengths = jnp.asarray(lengths, jnp.int32)
    edge = jnp.arange(N)[None, :] < lengths[:, None]            # blocks t = 1..n
    node = jnp.arange(N + 1)[None, :] <= lengths[:, None]       # prefixes h_0..h_n
    delta = jnp.where(edge, v - u + b[:, :-1] - b[:, 1:], 0.0)
    c = jnp.concatenate([jnp.zeros((B, 1), delta.dtype), jnp.cumsum(delta, -1)], -1)
    return dict(delta=delta, c=jnp.where(node, c, 0.0), edge=edge, node=node,
                lengths=lengths, n=lengths.astype(jnp.float32))


# ---- losses (per rollout, shape [B]) -----------------------------------------------------------

def local_loss(r):
    """mean_t delta_t^2."""
    return (r["delta"] ** 2).sum(-1) / r["n"]


def all_intervals_loss(r):
    """mean over every interval of Delta_{i,j}^2 = 2(n+1)/n * Var(c_0..c_n), population variance."""
    m = r["n"] + 1.0
    mean = r["c"].sum(-1) / m
    centred = jnp.where(r["node"], r["c"] - mean[:, None], 0.0)
    return 2.0 * m / r["n"] * (centred ** 2).sum(-1) / m


def scale(r):
    """s(n) = (n+2)/3: E[L_all] under iid zero-mean residuals, so L_all/s(n) sits at the L_local scale."""
    return (r["n"] + 2.0) / 3.0


def mixed_loss(r, alpha=0.5, normalize=True):
    """(1-alpha) * L_local + alpha * L_all / s(n)  (note section 3). normalize=False keeps raw L_all."""
    a = all_intervals_loss(r)
    return (1.0 - alpha) * local_loss(r) + alpha * (a / scale(r) if normalize else a)


def poly_loss(r, coefs=(1.0,)):
    """sum_{i<j} w(j-i) (c_j-c_i)^2 / sum_{i<j} w(j-i)  with w(l) = sum_d coefs[d] * l^d, in O(dn).

    coefs=(1.0,) is all_intervals_loss; coefs=(0.0, 1.0) weights each interval by its length, which tilts the
    objective toward long-range drift without going all the way to the multiscale machinery."""
    c, node, n = r["c"], r["node"], r["lengths"]
    B, M = c.shape
    f = c.dtype
    k = jnp.arange(M, dtype=f)
    c2 = c * c
    nodef = node.astype(f)
    zerosB = jnp.zeros((B, 1), f)
    num = jnp.zeros(B, f)
    den = jnp.zeros(B, f)
    for d, a in enumerate(coefs):
        if a == 0.0:
            continue
        for rr in range(d + 1):
            s, sign = d - rr, ((-1.0) ** (d - rr)) * comb(d, rr) * a
            jr, is_ = k ** rr, k ** s
            P_s_lt = jnp.concatenate([jnp.zeros(1, f), jnp.cumsum(is_)[:-1]])   # sum_{i<j} i^s
            P_r = jnp.cumsum(jr)                                                # sum_{j<=k} j^r
            A = (jr * c2 * P_s_lt).sum(-1)                                      # sum_{i<j} j^r i^s c_j^2
            Bt = (is_ * c2 * (P_r[n][:, None] - P_r[None, :])).sum(-1)          # ... c_i^2, j capped at n
            cs = jnp.concatenate([zerosB, jnp.cumsum(is_ * c, -1)[:, :-1]], -1)  # sum_{i<j} i^s c_i
            num = num + sign * (A + Bt - 2.0 * (jr * c * cs).sum(-1))
            den = den + sign * ((jr * P_s_lt)[None, :] * nodef).sum(-1)
    return num / jnp.maximum(den, 1e-30)


def multiscale_loss(r, divide_by_length=True, add_full=False):
    """Equal weight per power-of-two length; each squared interval residual optionally divided by its length
    (which equalizes them under the same iid heuristic). O(n log n)."""
    c, lengths = r["c"], r["lengths"]
    B, M = c.shape
    N = M - 1
    lens = [1]
    while lens[-1] * 2 <= N:
        lens.append(lens[-1] * 2)
    if add_full and N not in lens:
        lens.append(N)
    total = jnp.zeros(B, c.dtype)
    count = jnp.zeros(B, c.dtype)
    for ell in lens:
        valid = (jnp.arange(M - ell)[None, :] + ell) <= lengths[:, None]
        diff = jnp.where(valid, c[:, ell:] - c[:, :-ell], 0.0)
        terms = valid.sum(-1).astype(c.dtype)
        per = (diff ** 2).sum(-1) / jnp.maximum(terms, 1.0) / (ell if divide_by_length else 1.0)
        total = total + per
        count = count + (terms > 0).astype(c.dtype)
    return total / jnp.maximum(count, 1.0)


ALL = {
    "local": local_loss,
    "all": all_intervals_loss,
    "all_scaled": lambda r: all_intervals_loss(r) / scale(r),
    "mixed": mixed_loss,
    "poly_len": lambda r: poly_loss(r, (0.0, 1.0)),
    "multiscale": multiscale_loss,
}


def all_losses(r):
    """Every loss in ALL, each [B]."""
    return {k: fn(r) for k, fn in ALL.items()}


# ---- collapse diagnostics ----------------------------------------------------------------------

def diagnostics(r, u, v, b):
    """The degenerate optimum of every loss here is "ignore R": v_t == u_t and b_t constant in t makes each
    delta exactly zero. Watch these alongside the loss -- if they fall toward zero while the loss does,
    lambda_cons is too high (note section 6)."""
    edge, node, n = r["edge"], r["node"], r["n"]
    u, v, b = (jnp.asarray(z, jnp.float32) for z in (u, v, b))
    last = jnp.take_along_axis(b, r["lengths"][:, None], 1)[:, 0]
    return dict(
        info_gain=last - b[:, 0],                                       # log q_n(R) - log q_0(R), should grow
        cond_gap=(jnp.abs(v - u) * edge).sum(-1) / n,                   # mean |v_t - u_t|, R changing the policy
        b_first=b[:, 0], b_last=last,
        drift=r["c"][jnp.arange(b.shape[0]), r["lengths"]] / n,         # mean signed residual = c_n / n
        rms_delta=jnp.sqrt((r["delta"] ** 2).sum(-1) / n),
    )


# ---- brute force references (O(n^2); tests and by-length diagnostics) --------------------------

def weight_profile(n):
    """diag(A) for L_all as a quadratic form in delta: A[s,t] = min(s,t) * (n+1-max(s,t)), s,t = 1..n."""
    s = np.arange(1, n + 1)
    return s * (n + 1 - s)


def all_pairs_bruteforce(c_row, n, coefs=None):
    """Explicit mean over every interval of one rollout; coefs as in poly_loss (None = uniform)."""
    c = np.asarray(c_row, np.float64)[: n + 1]
    i, j = np.triu_indices(n + 1, 1)
    w = 1.0 if coefs is None else sum(a * (j - i) ** float(d) for d, a in enumerate(coefs))
    return float((w * (c[j] - c[i]) ** 2).sum() / np.sum(w * np.ones_like(i, float)))


def interval_stats_by_length(c, lengths, max_pairs=None):
    """RMS of Delta_{i,j} grouped by interval length, pooled over rollouts: arrays (length, rms, count).
    Pure diffusion of iid residuals gives rms ~ sqrt(l); coherent drift gives rms ~ l."""
    c, lengths = np.asarray(c, np.float64), np.asarray(lengths, np.int64)
    N = c.shape[1] - 1
    sq = np.zeros(N + 1)
    cnt = np.zeros(N + 1, np.int64)
    for row, n in zip(c, lengths):
        for ell in range(1, n + 1):
            d = row[ell : n + 1] - row[: n + 1 - ell]
            sq[ell] += (d ** 2).sum()
            cnt[ell] += d.size
    keep = cnt > 0
    ell = np.arange(N + 1)[keep]
    return ell, np.sqrt(sq[keep] / cnt[keep]), cnt[keep]


# ---- getting u, v, b out of the model ----------------------------------------------------------

def make_terms_fn(model, tok, jit=True):
    """(params, x_nor, x_R, targets, R_bin) -> dict of [B, T] / [B, T+1] arrays; jit'd unless jit=False
    (pass jit=False to call it inside another jit'd loss, as train.make_step does).

    x_nor and x_R are the same rollout bodies with MODE = NOR and MODE = R; targets comes from
    tok.next_targets (identical for both, it only reads the body). Nothing is detached: gradients flow into
    both orderings and into the NOR value head, per note section 6."""
    types = jnp.asarray(tok.types)
    sa = jnp.asarray(tok.sidx[: tok.T])          # slot that predicts step t's action
    sn = jnp.asarray(tok.sidx)                   # state slots, prefixes h_0..h_T

    def gather(next_logits, targets):
        lp = jax.nn.log_softmax(next_logits[:, :-1].astype(jnp.float32), -1)
        return jnp.take_along_axis(lp, jnp.maximum(targets, 0)[..., None], -1)[..., 0]

    def terms(params, x_nor, x_R, targets, R_bin):
        out_n = model.apply({"params": params}, x_nor, types)
        out_r = model.apply({"params": params}, x_R, types)
        lp_n, lp_r = gather(out_n["next"], targets), gather(out_r["next"], targets)
        B = x_nor.shape[0]
        logq = jax.nn.log_softmax(out_n["value"][:, sn].astype(jnp.float32), -1)      # [B, T+1, K]
        b = jnp.take_along_axis(logq, R_bin[:, None, None].repeat(sn.size, 1), -1)[..., 0]
        u_pi, u_dyn = lp_n[:, sa], lp_n[:, sa + 1]
        v_pi, v_dyn = lp_r[:, sa], lp_r[:, sa + 1]
        return dict(u=u_pi + u_dyn, v=v_pi + v_dyn, b=b,
                    u_pi=u_pi, u_dyn=u_dyn, v_pi=v_pi, v_dyn=v_dyn, logq=logq,
                    lp_nor=lp_n, lp_R=lp_r)      # every slot, so the data loss can reuse these two passes

    return jax.jit(terms) if jit else terms


def rollout_batch(tok, maze, d, idx):
    """Host-side inputs for one batch of rollouts: the NOR and R token arrays, shared targets, R bin, length."""
    body = tok.encode_body(d["positions"][idx], d["actions"][idx], d["length"][idx])
    R_bin = maze.outcome_bin(d["length"][idx], d["reached"][idx]).astype(np.int32)
    x_nor, x_R = tok.with_mode(body, None), tok.with_mode(body, R_bin)
    targets, _ = tok.next_targets(x_nor)
    return dict(x_nor=x_nor, x_R=x_R, targets=targets, R_bin=R_bin,
                lengths=d["length"][idx].astype(np.int32))


def data_nll(lp, mask):
    """Teacher-forced next-token loss from gathered log probs, matching model.next_token_loss."""
    m = mask.astype(lp.dtype)
    return -(lp * m).sum() / jnp.maximum(m.sum(), 1.0)


def evaluate(params, terms_fn, batch):
    """One batch end to end: model terms -> residuals -> every loss + diagnostics. All values per rollout."""
    t = terms_fn(params, *(jnp.asarray(batch[k]) for k in ("x_nor", "x_R", "targets", "R_bin")))
    r = residuals(t["u"], t["v"], t["b"], batch["lengths"])
    return t, r, all_losses(r), diagnostics(r, t["u"], t["v"], t["b"])


# ---- the exact model: what a perfectly consistent model scores ---------------------------------

def exact_terms(maze, gt, positions, actions, lengths, R_bins, n_max=None):
    """u, v, b for the TRUE joint distribution, from the DP -- the reference every loss here is measured
    against. All six objectives evaluate to exactly 0 on these, because

        v_t - u_t = log piR*(a_t | t, s_t, k) - log(1/4) = log child_h[t,s_t,a_t,k] - log h[t,s_t,k]

    and child_h[t,s,a,k] = h[t+1, next(s,a), k] = exp(b_{t+1}), so delta_t = b_{t+1} - b_t + b_t - b_{t+1} = 0.
    The dynamics are deterministic, so their log ratio contributes nothing on a valid path.

    Read against a model's own losses this makes them absolute: 0 is perfect consistency, not a floor that
    has to be estimated. Only the trajectory's own outcome bin is safe here -- h[t, s_t, k] is 0 for a bin the
    prefix has already ruled out, where the log form is undefined (note section 7, "handle exact zeros")."""
    positions, actions = np.asarray(positions), np.asarray(actions)
    lengths, R_bins = np.asarray(lengths).astype(np.int64), np.asarray(R_bins).astype(np.int64)
    N = len(lengths)
    T = maze.T if n_max is None else n_max
    t_ix = np.arange(T)
    edge = t_ix[None, :] < lengths[:, None]                   # blocks t = 0..n-1
    node = np.arange(T + 1)[None, :] <= lengths[:, None]      # prefixes h_0..h_n
    s = positions[:, :T].astype(np.int64)
    a = actions[:, :T].astype(np.int64)
    k = R_bins[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(gt.piR_star[t_ix[None, :], s, k, a])       # [N, T]
        b = np.log(gt.h[np.arange(T + 1)[None, :], positions[:, :T + 1].astype(np.int64), k])
    u = np.full((N, T), -np.log(N_ACTIONS_))                  # uniform behaviour policy, deterministic dynamics
    return (np.where(edge, u, 0.0), np.where(edge, np.nan_to_num(v), 0.0),
            np.where(node, np.nan_to_num(b), 0.0))


def exact_losses(maze, gt, d, idx):
    """Every consistency loss for the true model on rollouts `idx` of dataset `d`. All ~0."""
    u, v, b = exact_terms(maze, gt, d["positions"][idx], d["actions"][idx], d["length"][idx],
                          maze.outcome_bin(d["length"][idx], d["reached"][idx]))
    r = residuals(u, v, b, d["length"][idx])
    return all_losses(r), r


def make_heldout_eval(model, tok, maze, d, idx):
    """An eval_fn-compatible callable: every consistency loss plus the collapse diagnostics, on one fixed
    held-out batch. Unlike the `cons` value in the training log -- which is whichever objective that run
    optimizes, on its own moving batch -- these are the same losses on the same rollouts for every config,
    so they compare directly. The true model scores 0 on all of them (exact_losses)."""
    terms_fn = make_terms_fn(model, tok)
    batch = rollout_batch(tok, maze, d, idx)

    def fn(params):
        _, _, losses, diag = evaluate(params, terms_fn, batch)
        out = {f"cons/{k}": float(jnp.mean(v)) for k, v in losses.items()}
        out.update({f"diag/{k}": float(jnp.mean(diag[k])) for k in ("cond_gap", "info_gain", "drift")})
        return out

    return fn
