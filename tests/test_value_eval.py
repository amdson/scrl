import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from maze_consistency.dataset import canonical_maze
from maze_consistency.env import maze_from_ascii, random_walk_episodes
from maze_consistency.value_eval import hitting_value, expected_return_table, prefix_expected_return

M = canonical_maze()


def test_expected_return():
    V = expected_return_table(M)
    assert np.abs(V[0] - hitting_value(M)).max() <= M.gamma ** (M.T + 1)      # truncation costs E[gamma**tau; tau > T]
    m = maze_from_ascii(M.ascii(), T=40, gamma=M.gamma, n_bins=M.K)            # short horizon: truncation matters
    V = expected_return_table(m)
    d = random_walk_episodes(m, 40000, 0)
    ev = prefix_expected_return(m, d["positions"], d["length"], V)
    for t in (0, 10, 30):                                                     # E[R | prefix] is a martingale
        alive = d["length"] >= t
        res = d["returns"][alive] - ev[alive, t]
        assert abs(res.mean()) < 4 * res.std() / np.sqrt(alive.sum()) + 1e-9
    r = np.flatnonzero(d["reached"])[:100]
    assert np.allclose(ev[r, d["length"][r]], m.gamma ** d["length"][r])      # arrived: E[R] = gamma**L
    assert np.isnan(ev[r, -1][d["length"][r] < m.T]).all()
