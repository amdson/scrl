"""Spliced model rollouts: the prefix is kept, R is the achieved bin, and imagined dynamics read the model's
world model at the right slot -- a perfect world model reproduces the real maze exactly."""
import os
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from maze_consistency.dataset import load
from maze_consistency.tokens import Tokenizer
from maze_consistency.model import ModelConfig, MazeTransformer
from maze_consistency.dp import compute_ground_truth
from maze_consistency.evaluate import make_action_logits, make_cell_logits, continue_rollout, rollout_diagnostics
from maze_consistency.augment import splice, summarize, RolloutBuffer
from maze_consistency.train import train, LossConfig, N_HELDOUT

MAZE, D = load()
TOK = Tokenizer(MAZE)
CFG = dict(d_model=32, n_layers=1, n_heads=2)
MODEL = MazeTransformer(ModelConfig.for_tokenizer(TOK, **CFG))
P = MODEL.init(jax.random.PRNGKey(0), jnp.asarray(TOK.blank(1)), jnp.asarray(TOK.types))["params"]
AL = make_action_logits(MODEL, TOK)
POOL = np.arange(len(D["length"]) - N_HELDOUT)


class NoOracleMaze:
    """Fail immediately if generation or a training objective asks for exact maze knowledge."""
    def __getattr__(self, name):
        if name in ("dist", "next_open", "best_bin", "R_opt", "start_cells", "grid"):
            raise AssertionError(f"training accessed maze.{name}")
        return getattr(MAZE, name)


def diagnostics(ro, tau):
    return rollout_diagnostics(MAZE, dict(ro, tau=tau, achieved=ro["bins"],
                                         orig_bin=np.zeros(len(tau), dtype=int),
                                         requested=np.full(len(tau), MAZE.K - 1)))


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
    assert (diagnostics(ro, tau)["n_bad"] == 0).all()
    # the real (untrained) world model: prefix kept, and n_bad counts exactly the impossible moves after tau
    ro = continue_rollout(P, AL, *args, rng, cell_logits=make_cell_logits(MODEL, TOK))
    diag = diagnostics(ro, tau)
    for i in range(6):
        t0, L = int(tau[i]), int(ro["length"][i])
        assert (ro["positions"][i, :t0 + 1] == D["positions"][idx[i], :t0 + 1]).all()
        wrong = sum(ro["positions"][i, t + 1] != MAZE.next_open[ro["positions"][i, t], ro["actions"][i, t]]
                    for t in range(t0, L))
        assert wrong == diag["n_bad"][i], (i, wrong, diag["n_bad"][i])


def test_splice_relabels_and_asks_for_more():
    gt = compute_ground_truth(MAZE)
    rng = np.random.default_rng(1)
    for request in ("above", "highest", "best"):
        b = splice(P, AL, TOK, MAZE, D, POOL, 12, rng, tau_max=60, request=request, cell_logits=oracle_cells)
        assert (b["requested"] > b["orig_bin"]).all()                                    # always "higher"
        assert (b["achieved"] == MAZE.outcome_bin(b["length"], b["reached"])).all()      # relabelled
        assert (b["tau"] <= 60).all() and (b["length"] > b["tau"]).all()
        if request != "above":
            assert (b["requested"] == MAZE.K - 1).all()
        diag = rollout_diagnostics(MAZE, b, gt)
        assert ((diag["p_improve_rw"] >= 0) & (diag["p_improve_rw"] <= 1 + 1e-9)).all()
        s = summarize(MAZE, b, gt)
        assert 0 <= s["improved"] <= 1 and 0 <= s["improved_rw"] <= 1


def test_uniform_policy_matches_exact_random_walk_baseline():
    """End-to-end check on the whole splice pipeline. Swap the model for a uniform policy and the continuation
    IS a random walk, so the measured improve / got-request rates must match the DP's exact per-row baselines
    (p_improve_rw, p_request_rw) to within sampling error. Exercises prefix handling, the switch time, the real
    dynamics, relabelling, and the h[tau, s_tau] bookkeeping at once."""
    gt = compute_ground_truth(MAZE)
    uniform = lambda params, x, i: jnp.zeros((x.shape[0], 4))
    for request in ("above", "highest"):
        b = splice(None, uniform, TOK, MAZE, D, POOL, 800, np.random.default_rng(3), tau_max=100,
                   request=request, cell_logits=oracle_cells)
        diag = rollout_diagnostics(MAZE, b, gt)
        assert (diag["n_bad"] == 0).all()
        for hit, p in (((b["achieved"] > b["orig_bin"]), diag["p_improve_rw"]),
                       ((b["achieved"] >= b["requested"]), diag["p_request_rw"])):
            diff = hit.mean() - p.mean()
            se = np.sqrt((p * (1 - p)).mean() / len(p))
            assert abs(diff) < 4 * se + 1e-3, (float(hit.mean()), float(p.mean()), float(se))


def test_generation_needs_no_oracle_and_retains_infeasible_requests():
    maze = NoOracleMaze()
    uniform = lambda params, x, i: np.zeros((x.shape[0], 4))
    def stay(params, x, i):
        s = np.asarray(x[:, i - 1, 1] + x[:, i - 1, 2] * maze.W)
        return np.where(np.arange(maze.n_cells)[None] == s[:, None], 0.0, -1e9)

    for request in ("above", "highest"):
        b = splice(None, uniform, TOK, maze, D, POOL, 64, np.random.default_rng(19), tau_max=100,
                   request=request, cell_logits=stay)
        # This fake world never reaches the goal: failed generations must remain in the training buffer.
        assert len(b["length"]) == 64 and not b["reached"].any()
        assert (b["achieved"] == 0).all()
        pos = b["positions"][np.arange(64), b["tau"]]
        oracle_best = MAZE.success_bin(np.minimum(b["tau"] + MAZE.dist[pos], MAZE.T))
        assert (b["requested"] > oracle_best).any(), "requests must not be clipped to oracle feasibility"
        before = {k: v.copy() for k, v in b.items()}
        summarize(MAZE, b)
        for k in b:
            np.testing.assert_array_equal(b[k], before[k])


def test_training_refuses_environment_generation_and_oracle_objective():
    calls = [
        lambda: splice(None, None, TOK, MAZE, D, POOL, 1, np.random.default_rng(0)),
        lambda: RolloutBuffer(MODEL, TOK, MAZE, D, POOL, dynamics="env"),
        lambda: train(steps=1, loss=LossConfig(mc=True, a=True)),
    ]
    for call in calls:
        try:
            call()
        except ValueError as e:
            assert "real-maze" in str(e)
        else:
            raise AssertionError("oracle-dependent training should be refused")


def test_train_mixes_rollouts_without_oracles_and_refuses_td():
    out = tempfile.mkdtemp()
    maze = NoOracleMaze()
    buf = RolloutBuffer(MODEL, TOK, maze, D, POOL, size=8, refresh_every=2, start_after=1, tau_max=30,
                        log=lambda *a: None)
    # Only the evaluation callback receives the real maze; generation and losses receive the guarded one.
    with patch("maze_consistency.train.load", return_value=(maze, D)), \
         patch("maze_consistency.augment.summarize", side_effect=lambda _, b: summarize(MAZE, b)):
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
