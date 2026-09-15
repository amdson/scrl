"""Spliced model rollouts: the prefix is kept, R is the achieved bin, and imagined dynamics read the model's
world model at the right slot -- a perfect world model reproduces the real maze exactly."""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from maze_consistency.dataset import load
from maze_consistency.tokens import Tokenizer
from maze_consistency.model import ModelConfig, MazeTransformer
from maze_consistency.dp import compute_ground_truth
from maze_consistency.evaluate import make_action_logits, make_cell_logits, continue_rollout
from maze_consistency.augment import splice, summarize, RolloutBuffer
from maze_consistency.train import train, LossConfig, N_HELDOUT

MAZE, D = load()
TOK = Tokenizer(MAZE)
CFG = dict(d_model=32, n_layers=1, n_heads=2)
MODEL = MazeTransformer(ModelConfig.for_tokenizer(TOK, **CFG))
P = MODEL.init(jax.random.PRNGKey(0), jnp.asarray(TOK.blank(1)), jnp.asarray(TOK.types))["params"]
AL = make_action_logits(MODEL, TOK)
POOL = np.arange(len(D["length"]) - N_HELDOUT)


def test_continue_rollout_keeps_prefix_and_real_dynamics():
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(D["length"][POOL] > 30), 6)
    tau = np.array([0, 1, 5, 12, 20, 29])
    ro = continue_rollout(P, AL, TOK, MAZE, D["positions"][idx], D["actions"][idx], tau,
                          np.full(6, MAZE.K - 1), rng)
    for i in range(6):
        t0, L = int(tau[i]), int(ro["length"][i])
        assert (ro["positions"][i, :t0 + 1] == D["positions"][idx[i], :t0 + 1]).all()     # prefix verbatim
        assert (ro["actions"][i, :t0] == D["actions"][idx[i], :t0]).all()
        assert L > t0
        for t in range(L):                                                               # the real maze
            assert ro["positions"][i, t + 1] == MAZE.next_open[ro["positions"][i, t], ro["actions"][i, t]]
        assert bool(ro["reached"][i]) == (ro["positions"][i, L] == MAZE.goal)


def oracle_cells(params, x, i):
    """A perfect world model: one-hot on maze.next_open for the state at slot i-1 and the action at slot i."""
    x = np.asarray(x)
    s = np.minimum(x[:, i - 1, 1] + x[:, i - 1, 2] * MAZE.W, MAZE.n_cells - 1)
    a = np.clip(x[:, i, 0] - TOK.ACT0, 0, 3)
    lg = np.full((x.shape[0], MAZE.n_cells), -1e9)
    lg[np.arange(len(s)), MAZE.next_open[s, a]] = 0.0
    return lg


def test_imagined_dynamics_read_the_right_slots_and_count_hallucinations():
    rng = np.random.default_rng(4)
    idx = rng.choice(np.flatnonzero(D["length"][POOL] > 30), 6)
    tau = np.array([0, 3, 8, 15, 22, 29])
    args = (TOK, MAZE, D["positions"][idx], D["actions"][idx], tau, np.full(6, MAZE.K - 1))
    # a perfect world model must reproduce the real maze: no impossible move anywhere
    ro = continue_rollout(P, AL, *args, rng, cell_logits=oracle_cells)
    assert (ro["n_bad"] == 0).all()
    # the real (untrained) world model: prefix kept, and n_bad counts exactly the impossible moves after tau
    ro = continue_rollout(P, AL, *args, rng, cell_logits=make_cell_logits(MODEL, TOK))
    for i in range(6):
        t0, L = int(tau[i]), int(ro["length"][i])
        assert (ro["positions"][i, :t0 + 1] == D["positions"][idx[i], :t0 + 1]).all()
        wrong = sum(ro["positions"][i, t + 1] != MAZE.next_open[ro["positions"][i, t], ro["actions"][i, t]]
                    for t in range(t0, L))
        assert wrong == ro["n_bad"][i], (i, wrong, ro["n_bad"][i])


def test_splice_relabels_and_asks_for_more():
    gt = compute_ground_truth(MAZE)
    rng = np.random.default_rng(1)
    for request in ("above", "best"):
        b = splice(P, AL, TOK, MAZE, D, POOL, 12, rng, tau_max=60, request=request, gt=gt)
        assert (b["requested"] > b["orig_bin"]).all()                                    # always "higher"
        assert (b["achieved"] == MAZE.outcome_bin(b["length"], b["reached"])).all()      # relabelled
        assert (b["tau"] <= 60).all() and (b["length"] > b["tau"]).all()
        assert ((b["p_improve_rw"] >= 0) & (b["p_improve_rw"] <= 1 + 1e-9)).all()
        s = summarize(MAZE, b)
        assert 0 <= s["improved"] <= 1 and 0 <= s["improved_rw"] <= 1


def test_uniform_policy_matches_exact_random_walk_baseline():
    """End-to-end check on the whole splice pipeline. Swap the model for a uniform policy and the continuation
    IS a random walk, so the measured improve / got-request rates must match the DP's exact per-row baselines
    (p_improve_rw, p_request_rw) to within sampling error. Exercises prefix handling, the switch time, the real
    dynamics, relabelling, and the h[tau, s_tau] bookkeeping at once."""
    gt = compute_ground_truth(MAZE)
    uniform = lambda params, x, i: jnp.zeros((x.shape[0], 4))
    for cells in (None, oracle_cells):                       # the real maze, and a perfect imagined one
        b = splice(None, uniform, TOK, MAZE, D, POOL, 800, np.random.default_rng(3), tau_max=100, gt=gt,
                   cell_logits=cells)
        assert (b["n_bad"] == 0).all()
        for hit, p in (((b["achieved"] > b["orig_bin"]), b["p_improve_rw"]),
                       ((b["achieved"] >= b["requested"]), b["p_request_rw"])):
            diff = hit.mean() - p.mean()
            se = np.sqrt((p * (1 - p)).mean() / len(p))
            assert abs(diff) < 4 * se + 1e-3, (float(hit.mean()), float(p.mean()), float(se))


def test_train_mixes_rollouts_and_refuses_td():
    out = tempfile.mkdtemp()
    buf = RolloutBuffer(MODEL, TOK, MAZE, D, POOL, size=8, refresh_every=2, start_after=1, tau_max=30,
                        log=lambda *a: None)
    train(name=out, steps=3, batch=8, loss=LossConfig(mc=True, cons=True, cons_loss="local", cons_batch=4),
          mixer=buf, mix_frac=0.5, log=lambda *a: None, **CFG)
    assert [h["step"] for h in buf.history] == [1, 3]                                     # refresh schedule
    try:
        train(name=out, steps=1, loss=LossConfig(td=True), mixer=buf, mix_frac=0.5, log=lambda *a: None, **CFG)
    except ValueError:
        pass
    else:
        raise AssertionError("td with model-generated rows should be refused")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
