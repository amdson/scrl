import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp

from maze_consistency.env import get_maze, random_walk_episodes, dp_state, rollout_numpy, table_policy
from maze_consistency.dp import compute_ground_truth, check_identities
from maze_consistency.tokens import Tokenizer, r_distribution
from maze_consistency.train import TrainConfig, build, make_train_step, make_batch
from maze_consistency.losses import LossConfig
from maze_consistency.eval import EvalSet, evaluate_model


def test_dp_identities():
    for name in ("default", "door"):
        m = get_maze(name)
        gt = compute_ground_truth(m)
        c = check_identities(gt)
        assert c["B_ok"], c
        assert c["piR_sum_err"] < 1e-9
        assert c["h0_Rmax"] >= 1e-3, f"{name}: h0={c['h0_Rmax']}"
        assert np.allclose(gt.h.sum(-1), 1.0)
        assert np.allclose(gt.d.sum(-1)[0], 1.0)


def test_empirical_matches_dp():
    m = get_maze("door")
    gt = compute_ground_truth(m)
    d = random_walk_episodes(m, 40000, 0)
    emp = np.bincount(d["returns"] + m.T + 1, minlength=m.K) / d["N"]
    assert np.abs(emp - gt.h[0, m.start]).max() < 0.01
    # optimal policy reaches goal in d* steps when the door is open
    ro = rollout_numpy(m, table_policy(gt.policy_piR(m.R_max), m), 2000, np.random.default_rng(0))
    assert (ro["returns"][~ro["locked"]] == m.R_max).all()


def test_tokenizer_roundtrip():
    m = get_maze("door")
    tok = Tokenizer(m)
    d = random_walk_episodes(m, 50, 1)
    t = tok.encode(d, R_bin=None)
    assert t.shape == (50, tok.L)
    assert (t[:, 0] == tok.NOR).all()
    for i in range(50):
        L = d["length"][i]
        assert (t[i, 1:2 * L + 1:2] == d["actions"][i, :L]).all()
        assert (t[i, 2 * L + 1:] == tok.PAD).all()


def test_train_step_and_eval():
    cfg = TrainConfig(d_model=32, n_layers=1, n_heads=2, batch=16, N=500, maze="door",
                      closure_softq=True, closure_distill=True)
    maze, gt, tok, model, state, fwd = build(cfg)
    data = random_walk_episodes(maze, 500, 0)
    rng = np.random.default_rng(0)
    lc = LossConfig.named("TDA", td_backup="piR", distill=True)
    step = make_train_step(model, tok, lc, True, True)
    b = make_batch(data, rng.integers(0, data["N"], 16), tok, rng, r_distribution(maze), int(maze.bin_of(maze.R_max)), True)
    state2, losses = step(state, b, state.params)
    assert np.isfinite(float(losses["total"]))
    assert all(np.isfinite(float(v)) for v in losses.values())
    es = EvalSet(maze, gt, tok, n=100, n_opt=32)
    m = evaluate_model(state2.params, fwd, es, rollouts=True, rollout_n=16, fqi=True)
    assert len(m["value_err"]) == maze.T + 1
    assert 0 <= m["rollouts"]["posterior"]["door_choice"] <= 1


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
