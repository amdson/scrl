import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from maze_consistency.dataset import canonical_maze
from maze_consistency.env import random_walk_episodes
from maze_consistency.dp import compute_ground_truth, check_identities
from maze_consistency.tokens import Tokenizer

M = canonical_maze()
GT = compute_ground_truth(M)


def test_dp():
    c = check_identities(GT)
    assert c["B_residual"] < 1e-12 and c["piR_sum_err"] < 1e-9
    assert np.allclose(GT.h.sum(-1), 1.0)
    assert np.allclose(GT.V_opt[0, M.start_cells], M.gamma ** M.dist[M.start_cells])   # optimal = gamma**dist


def test_bins_and_returns():
    L = np.arange(M.T + 1)
    b = M.success_bin(L)
    assert b.min() == 1 and b.max() == M.K - 1 and (np.diff(b) <= 0).all()   # slower arrival -> lower bin
    assert M.outcome_bin(5, False) == M.FAIL_BIN
    R, e = M.gamma ** L.astype(float), M.bin_edges
    assert ((R >= e[b - 1] * (1 - 1e-12)) & ((R < e[np.minimum(b, M.K - 1)]) | (b == M.K - 1))).all()
    d = random_walk_episodes(M, 20000, 1)
    assert np.isin(d["positions"][:, 0], M.start_cells).all()
    assert np.allclose(d["returns"], M.return_of(d["length"], d["reached"]))
    emp = np.bincount(M.outcome_bin(d["length"], d["reached"]), minlength=M.K) / d["N"]
    assert np.abs(emp - GT.h[0, M.start_cells].mean(0)).max() < 0.01          # sampler matches the exact DP


def test_tokens():
    tok = Tokenizer(M)
    d = random_walk_episodes(M, 50, 2)
    t = tok.encode(d)
    assert t.shape == (50, tok.L, 3) and (t[:, 0, 0] == tok.NOR).all()
    for i in range(50):
        L = d["length"][i]
        assert (t[i, 1] == tok.pos(d["positions"][i, 0])).all()                  # explicit start
        acts, poss, cells = t[i, 2:2 * L + 2:2], t[i, 3:2 * L + 2:2], d["positions"][i, 1:L + 1]
        assert (acts[:, 0] == d["actions"][i, :L]).all() and (acts[:, 1] == tok.XN).all()
        assert (poss[:, 1] == cells % M.W).all() and (poss[:, 2] == cells // M.W).all()
        assert (t[i, 2 * L + 2:] == tok.pad_slot).all()
    assert t[..., 0].max() < tok.n_kind and t[..., 1].max() < tok.n_x and t[..., 2].max() < tok.n_y


def test_next_targets():
    tok = Tokenizer(M)
    d = random_walk_episodes(M, 50, 3)
    tgt, mask = tok.next_targets(tok.encode(d))
    for i in range(50):
        L = d["length"][i]
        assert tgt[i, 0] == tok.OUT_CELL0 + d["positions"][i, 0]                              # MODE -> start
        assert (tgt[i, tok.sidx[:L]] == d["actions"][i, :L]).all()                          # state slot -> action
        assert (tgt[i, tok.aidx[:L]] == tok.OUT_CELL0 + d["positions"][i, 1:L + 1]).all()   # action slot -> cell
        assert mask[i].sum() == min(2 * L + 2, tok.L - 1)
        if L < M.T:
            assert tgt[i, 2 * L + 1] == tok.OUT_END                                         # goal -> END
    assert tgt[mask].min() >= 0 and tgt[mask].max() < tok.n_out


def test_binning_schemes():
    """Both schemes bin arrival time monotonically; geometric spends far more of its bins on the range
    optimal play can actually reach, which is the whole point of it."""
    import numpy as _np
    from maze_consistency.dataset import canonical_maze
    L = _np.arange(1, M.T + 1)
    used = {}
    for binning in ("uniform", "geometric"):
        for K in (6, 12, 24):
            m = canonical_maze(n_bins=K, binning=binning)
            b = m.success_bin(L)
            assert (_np.diff(b) <= 0).all(), (binning, K)                  # never faster -> lower bin
            assert b.min() >= 1 and b.max() <= K - 1
            assert m.outcome_bin(_np.array([5]), _np.array([False]))[0] == m.FAIL_BIN
            used[(binning, K)] = len(set(int(x) for x in m.success_bin(_np.arange(1, m.dist.max() + 1))))
    for K in (6, 12, 24):
        assert used[("geometric", K)] > used[("uniform", K)], K
    assert used[("uniform", 12)] == 2 and used[("geometric", 12)] == 7     # 2 of 11 bins vs 7 of 11

    assert canonical_maze(n_bins=12, binning="uniform").empty_bins == []
    assert canonical_maze(n_bins=24, binning="geometric").empty_bins == [18, 21, 22]

    # the default is unchanged, so every existing run and cached test set stays valid
    d = canonical_maze()
    assert d.binning == "uniform" and d.K == 12
    assert (d.success_bin(L) == (1 + (11 * (M.T - L)) // M.T).clip(1, 11)).all()


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
