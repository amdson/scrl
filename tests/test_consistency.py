"""The fast interval losses against explicit enumeration, in values and in gradients (note section 8)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from maze_consistency.dataset import canonical_maze
from maze_consistency.env import random_walk_episodes
from maze_consistency.tokens import Tokenizer
from maze_consistency.model import ModelConfig, MazeTransformer
import maze_consistency.consistency as C

M = canonical_maze()
RNG = np.random.default_rng(0)
LENGTHS = np.array([1, 2, 3, 7, 12, 5])          # includes n=1 and the padded max
N = 12


def _fake(B=len(LENGTHS), n=N):
    return RNG.normal(size=(B, n)), RNG.normal(size=(B, n)), RNG.normal(size=(B, n + 1))


def test_delta_matches_definition():
    u, v, b = _fake()
    c = np.asarray(C.residuals(u, v, b, LENGTHS)["c"])
    for row, n in enumerate(LENGTHS):
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                want = (v[row, i:j] - u[row, i:j]).sum() + b[row, i] - b[row, j]
                assert np.isclose(want, c[row, j] - c[row, i], atol=1e-5)


def test_fast_forms_match_enumeration():
    u, v, b = _fake()
    r = C.residuals(u, v, b, LENGTHS)
    c, d = np.asarray(r["c"]), np.asarray(r["delta"])
    bf = np.array([C.all_pairs_bruteforce(c[i], n) for i, n in enumerate(LENGTHS)])
    assert np.allclose(C.all_intervals_loss(r), bf, rtol=1e-4)
    assert np.allclose(C.poly_loss(r, (1.0,)), bf, rtol=1e-4)                       # uniform poly == all
    bf_len = np.array([C.all_pairs_bruteforce(c[i], n, (0.0, 1.0)) for i, n in enumerate(LENGTHS)])
    assert np.allclose(C.poly_loss(r, (0.0, 1.0)), bf_len, rtol=1e-4)               # length-weighted
    assert np.allclose(C.local_loss(r), [(d[i, :n] ** 2).sum() / n for i, n in enumerate(LENGTHS)], rtol=1e-5)
    ms = []
    for i, n in enumerate(LENGTHS):
        tot, cnt, ell = 0.0, 0, 1
        while ell <= N:
            if ell <= n:
                dd = c[i, ell:n + 1] - c[i, :n + 1 - ell]
                tot, cnt = tot + (dd ** 2).mean() / ell, cnt + 1
            ell *= 2
        ms.append(tot / max(cnt, 1))
    assert np.allclose(C.multiscale_loss(r), ms, rtol=1e-4)


def test_gradients_match_enumeration():
    """The variance shortcut must reproduce the fully differentiated squared pair loss, not just its value."""
    for n in (1, 3, 7):
        u, v, b = _fake(B=1)
        ln = np.array([n])

        def explicit(U):
            c = C.residuals(U, v, b, ln)["c"][0]
            return sum((c[j] - c[i]) ** 2 for i in range(n + 1) for j in range(i + 1, n + 1)) / (n * (n + 1) / 2)

        fast = jax.grad(lambda U: C.all_intervals_loss(C.residuals(U, v, b, ln))[0])(u)
        assert np.allclose(fast, jax.grad(explicit)(u), atol=1e-5)


def test_padding_and_masking():
    """Padded entries must never reach a loss: garbage past each length changes nothing."""
    u, v, b = _fake()
    r = C.residuals(u, v, b, LENGTHS)
    u2, v2, b2 = u.copy(), v.copy(), b.copy()
    for i, n in enumerate(LENGTHS):
        u2[i, n:], v2[i, n:], b2[i, n + 1:] = 1e3, -1e3, 1e3
    r2 = C.residuals(u2, v2, b2, LENGTHS)
    a, a2 = C.all_losses(r), C.all_losses(r2)
    for k in a:
        assert np.allclose(a[k], a2[k], rtol=1e-5), k


def test_weight_profile():
    n = 9
    A = np.array([[min(s, t) * (n + 1 - max(s, t)) for t in range(1, n + 1)] for s in range(1, n + 1)], float)
    assert (C.weight_profile(n) == np.diag(A)).all()
    d = RNG.normal(size=n)
    c = np.concatenate([[0.0], np.cumsum(d)])
    pairs = sum((c[j] - c[i]) ** 2 for i in range(n + 1) for j in range(i + 1, n + 1))
    assert np.isclose(d @ A @ d, pairs)


def test_known_cases():
    """Constant residual eps: L_local = eps^2 and L_all = (n+1)(n+2)eps^2/6 (note section 3)."""
    eps, n = 0.3, 9
    z = np.zeros((1, N))
    u, b = z, np.zeros((1, N + 1))
    v = np.where(np.arange(N)[None] < n, eps, 0.0)
    r = C.residuals(u, v, b, np.array([n]))
    assert np.isclose(C.local_loss(r)[0], eps ** 2, rtol=1e-5)
    assert np.isclose(C.all_intervals_loss(r)[0], (n + 1) * (n + 2) * eps ** 2 / 6, rtol=1e-4)
    # iid residuals: E[L_all] = (n+2)/3 * sigma^2, which is what scale() divides out
    assert np.isclose(C.scale(r)[0], (n + 2) / 3)


def test_zero_residual_at_exact_consistency():
    u, b = RNG.normal(size=(3, N)), RNG.normal(size=(3, N + 1))
    v = u + b[:, 1:] - b[:, :-1]                      # delta identically 0 by construction
    r = C.residuals(u, v, b, np.array([4, 9, 12]))
    for k, val in C.all_losses(r).items():
        assert np.allclose(val, 0.0, atol=1e-8), k


def test_model_terms_alignment():
    """u, v, b must line up with the tokenizer's slots, and the two passes must differ only via MODE."""
    tok = Tokenizer(M)
    d = random_walk_episodes(M, 6, seed=3)
    cfg = ModelConfig.for_tokenizer(tok, d_model=32, n_layers=1, n_heads=2)
    model = MazeTransformer(cfg)
    p = model.init(jax.random.PRNGKey(0), jnp.asarray(tok.blank(1)), jnp.asarray(tok.types))["params"]
    batch = C.rollout_batch(tok, M, d, np.arange(6))
    assert (batch["x_nor"][:, 0, 0] == tok.NOR).all()
    assert (batch["x_nor"][:, 1:] == batch["x_R"][:, 1:]).all()          # same body, only MODE differs
    t, r, losses, diag = C.evaluate(p, C.make_terms_fn(model, tok), batch)

    # u is the NOR pass's own log p of the action and the resulting cell, gathered from the full flat softmax
    out = model.apply({"params": p}, jnp.asarray(batch["x_nor"]), jnp.asarray(tok.types))
    lp = jax.nn.log_softmax(np.asarray(out["next"], np.float64), -1)
    logq = jax.nn.log_softmax(np.asarray(out["value"], np.float64), -1)
    for i in range(6):
        n = int(batch["lengths"][i])
        for tt in RNG.choice(n, size=min(n, 4), replace=False):
            want = (lp[i, tok.sidx[tt], d["actions"][i, tt]]
                    + lp[i, tok.aidx[tt], tok.OUT_CELL0 + d["positions"][i, tt + 1]])
            assert np.isclose(np.asarray(t["u"])[i, tt], want, atol=1e-4)
        for k in (0, n):
            assert np.isclose(np.asarray(t["b"])[i, k], logq[i, tok.sidx[k], batch["R_bin"][i]], atol=1e-4)
    assert all(np.isfinite(np.asarray(v)).all() for v in losses.values())
    assert all(np.isfinite(np.asarray(v)).all() for v in diag.values())


def test_exact_model_is_perfectly_consistent():
    """The strongest check on the whole construction: for the TRUE joint distribution every interval residual
    is identically 0, so every loss here is 0. If the block boundaries, the slot indexing, or the choice to
    gather from the full flat softmax were wrong, this is what would catch it."""
    from maze_consistency.dp import compute_ground_truth
    from maze_consistency.env import random_walk_episodes

    gt = compute_ground_truth(M)
    d = random_walk_episodes(M, 128, seed=11)
    idx = np.arange(128)
    losses, r = C.exact_losses(M, gt, d, idx)

    assert d["reached"][idx].any() and (~d["reached"][idx]).any()      # both arrivals and timeouts present
    assert d["length"][idx].min() < d["length"][idx].max()             # a spread of lengths
    assert np.abs(np.asarray(r["delta"])).max() < 1e-5                 # float32 epsilon on O(1) log probs
    for k, v in losses.items():
        assert np.asarray(v).max() < 1e-10, (k, float(np.asarray(v).max()))

    # and it is a real check, not one that passes on anything: perturbing any single term breaks it
    u, v, b = C.exact_terms(M, gt, d["positions"][idx], d["actions"][idx], d["length"][idx],
                            M.outcome_bin(d["length"][idx], d["reached"][idx]))
    for name, bump in (("u", (u + 0.01, v, b)), ("v", (u, v + 0.01, b)), ("b", (u, v, b + 0.01))):
        bad = C.all_losses(C.residuals(*bump, d["length"][idx]))
        if name == "b":
            continue        # a constant shift in b telescopes away, which is itself the identity working
        assert np.asarray(bad["local"]).max() > 1e-6, name


def test_lossconfig_validates_cons_loss():
    from maze_consistency.train import LossConfig
    LossConfig(cons=True, cons_loss="all_scaled")                 # every key of C.ALL is accepted
    for k in C.ALL:
        LossConfig(cons=True, cons_loss=k)
    try:
        LossConfig(cons=True, cons_loss="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("bad cons_loss should raise")
    assert LossConfig(cons=False, cons_loss="nope").cons_loss == "nope"    # unchecked when cons is off


def test_cons_training_step_runs_and_logs():
    """Each objective trains, and the collapse diagnostics are logged alongside it."""
    import jax.numpy as jnp
    import optax
    from maze_consistency.train import LossConfig, make_step, make_batch, value_batch, cons_batch
    from maze_consistency.env import random_walk_episodes
    from maze_consistency.model import ModelConfig, MazeTransformer

    tok = Tokenizer(M)
    d = random_walk_episodes(M, 8, seed=5)
    cfg = ModelConfig.for_tokenizer(tok, d_model=32, n_layers=1, n_heads=2)
    model = MazeTransformer(cfg)
    p0 = model.init(jax.random.PRNGKey(0), jnp.asarray(tok.blank(1)), jnp.asarray(tok.types))["params"]
    idx = np.arange(8)
    x, tgt, mask = make_batch(tok, M, d, idx)
    nb = cons_batch(tok, M, d, idx[:4])
    to_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)

    for key in C.ALL:
        lc = LossConfig(mc=True, cons=True, cons_loss=key, cons_batch=4)
        opt = optax.adam(1e-3)
        step = make_step(model, tok, opt, lc)
        _, _, parts = step(p0, opt.init(p0), jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask),
                           to_j(value_batch(M, d, idx[:4])), to_j(nb), jnp.float32(1.0))
        assert {"tf", "mc", "cons", "cond_gap", "info_gain"} <= set(parts), (key, sorted(parts))
        assert all(np.isfinite(float(v)) for v in parts.values()), key

    # cons_warmup gate: cons_on = 0 leaves the update identical to the same config with no consistency term
    lc = LossConfig(mc=True, cons=True, cons_loss="all_scaled", cons_batch=4)
    opt = optax.adam(1e-3)
    args = (jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(mask), to_j(value_batch(M, d, idx[:4])))
    off, _, _ = make_step(model, tok, opt, lc)(p0, opt.init(p0), *args, to_j(nb), jnp.float32(0.0))
    plain, _, _ = make_step(model, tok, opt, LossConfig(mc=True))(p0, opt.init(p0), *args, {}, jnp.float32(0.0))
    for a, b in zip(jax.tree_util.tree_leaves(off), jax.tree_util.tree_leaves(plain)):
        assert np.allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)


def test_starts_restrict_to_the_imagined_segment():
    """residuals(..., starts) on a row equals residuals on the sub-row [start, length] shifted to index 0, for
    every loss and the diagnostics: the prefix before `starts` is context only."""
    u, v, b = _fake()
    lengths, starts = np.array([4, 6, 12, 9, 12, 5]), np.array([0, 2, 5, 3, 11, 4])
    r = C.residuals(u, v, b, lengths, starts)
    losses = C.all_losses(r)
    dg = C.diagnostics(r, u, v, b)
    for row, (n, s) in enumerate(zip(lengths, starts)):
        u2, v2, b2 = np.zeros((1, N)), np.zeros((1, N)), np.zeros((1, N + 1))
        u2[0, : n - s], v2[0, : n - s], b2[0, : n - s + 1] = u[row, s:n], v[row, s:n], b[row, s : n + 1]
        r2 = C.residuals(u2, v2, b2, np.array([n - s]))
        for k, val in C.all_losses(r2).items():
            assert np.isclose(float(losses[k][row]), float(val[0]), rtol=1e-4, atol=1e-5), (k, row)
        dg2 = C.diagnostics(r2, u2, v2, b2)
        for k in ("info_gain", "cond_gap", "drift", "rms_delta"):
            assert np.isclose(float(dg[k][row]), float(dg2[k][0]), rtol=1e-4, atol=1e-5), (k, row)
    # the context blocks carry no gradient
    g = jax.grad(lambda U: C.all_intervals_loss(C.residuals(U, v, b, lengths, starts)).sum())(jnp.asarray(u))
    assert np.allclose(np.asarray(g)[np.arange(N)[None, :] < starts[:, None]], 0.0)


def test_quantile_query_is_the_most_ambitious_believed_threshold():
    tok = Tokenizer(M, cond="threshold")
    K = tok.K
    q = np.full((3, K), 1e-6)
    q[0, [1, 2, 3]] = [0.5, 0.3, 0.15]          # tails: k=1 ~.95, k=2 ~.45, k=3 ~.15, k=4 ~1e-5
    q[1, 0] = 1.0                                # nothing believed: falls back to k = 1
    q[2, K - 1] = 1.0                            # everything in the top bin: asks for K-1
    logq = np.log(q / q.sum(-1, keepdims=True))
    assert C._quantile_query(logq, 0.1, tok).tolist() == [3, 1, K - 1]
    assert C._quantile_query(logq, 0.4, tok).tolist() == [2, 1, K - 1]
    assert C._quantile_query(logq, 1e-4, tok).tolist() == [3, 1, K - 1]
    assert C._quantile_query(logq, 1e-7, tok).tolist() == [K - 1, K - 1, K - 1]
