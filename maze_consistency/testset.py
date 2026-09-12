"""An exact test set: sequences whose true next-token and value distributions are known from the DP.

Settings (rows): "NOR" random walks; "bin k" (k = 0..K-1) trajectories of the exact R-conditioned process
(start ~ P(start | k), actions ~ pi_R*(. | t, s, k)); "best far" starts >= FAR steps from the goal conditioned
on each start's own best bin, i.e. the near-optimal behaviour the training data never shows.

Stored targets (everything else is determined by the tokens):
    act_probs   [N, T, 4]      true action distribution at each state slot (NOR: 1/4; R: pi_R*), 0 after the end
    value       [N, T+1, K]    true V_t(. | s_t) = h[t, s_t] of the prefix read in NOR mode, 0 after the end
    start_nor   [n_cells]      true next-token distribution at a NOR mode slot (uniform over start cells)
    start_R     [K, n_cells]   true next-token distribution at an R mode slot, P(start | k) with a uniform prior
    next cell   one-hot at the recorded next position (deterministic dynamics); END after reaching the goal
"""
from __future__ import annotations

import os

import numpy as np

from .env import N_ACTIONS
from .dp import compute_ground_truth
from .tokens import Tokenizer

FAR = 10
TEST_PATH = "data/canonical/testset.npz"


def exact_rollouts(maze, gt, starts, bins, rng):
    """bins [N]: -1 = NOR (uniform random walk), else sample from pi_R*(. | t, s, bin)."""
    N, T = len(starts), maze.T
    pos = np.asarray(starts, dtype=np.int64).copy()
    positions = np.zeros((N, T + 1), dtype=np.int64)
    positions[:, 0] = pos
    actions = np.zeros((N, T), dtype=np.int64)
    act_probs = np.zeros((N, T, N_ACTIONS))
    length = np.full(N, T, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    r = bins >= 0
    for t in range(T):
        p = np.full((N, N_ACTIONS), 1.0 / N_ACTIONS)
        p[r] = gt.piR_star[t, pos[r], bins[r]]
        assert np.isfinite(p[alive]).all(), "conditioned process left the support"
        p = np.where(alive[:, None], p, 0.0)
        act_probs[:, t] = p
        a = np.minimum((rng.random(N)[:, None] > np.cumsum(np.where(alive[:, None], p, 0.25), -1)).sum(-1), N_ACTIONS - 1)
        actions[:, t] = a
        pos = np.where(alive, maze.next_open[pos, a], pos)
        positions[:, t + 1] = pos
        arrived = alive & (pos == maze.goal)
        length[arrived] = t + 1
        alive &= ~arrived
    return dict(positions=positions, actions=actions, length=length, reached=~alive, act_probs=act_probs)


def build_testset(maze, n_per=200, seed=7, path=TEST_PATH, log=print):
    gt = compute_ground_truth(maze)
    tok = Tokenizer(maze)
    rng = np.random.default_rng(seed)
    cells = maze.start_cells
    start_nor = np.zeros(maze.n_cells)
    start_nor[cells] = 1.0 / len(cells)
    start_R = np.zeros((maze.K, maze.n_cells))
    start_R[:, cells] = gt.h[0, cells].T
    start_R /= start_R.sum(1, keepdims=True)
    names = ["NOR"] + [f"bin {k}" for k in range(maze.K)] + ["best far"]
    starts, bins, setting = [], [], []
    for i, name in enumerate(names):
        if name == "NOR":
            s, b = rng.choice(cells, n_per), np.full(n_per, -1)
        elif name == "best far":
            s = rng.choice(cells[maze.dist[cells] >= FAR], n_per)
            b = maze.best_bin(s)
        else:
            k = i - 1
            s, b = rng.choice(maze.n_cells, n_per, p=start_R[k]), np.full(n_per, k)
        starts.append(s); bins.append(b); setting.append(np.full(n_per, i))
    starts, bins, setting = np.concatenate(starts), np.concatenate(bins), np.concatenate(setting)
    ro = exact_rollouts(maze, gt, starts, bins, rng)
    achieved = maze.outcome_bin(ro["length"], ro["reached"])
    r = bins >= 0
    assert (achieved[r] == bins[r]).all(), "exact conditioned rollouts must land in the requested bin"
    data = dict(positions=ro["positions"], actions=ro["actions"], length=ro["length"])
    tokens = tok.with_mode(tok.encode_body(data["positions"], data["actions"], data["length"]), np.maximum(bins, 0))
    tokens[~r, 0] = tok.mode(None)
    T = maze.T
    alive = np.arange(T + 1)[None] <= ro["length"][:, None]
    value = np.where(alive[..., None], gt.h[np.arange(T + 1)[None], ro["positions"]], 0.0)
    out = dict(tokens=tokens.astype(np.int16), setting=setting, setting_names=np.array(names), mode_bin=bins,
               positions=ro["positions"].astype(np.int16), actions=ro["actions"].astype(np.int8),
               length=ro["length"].astype(np.int16), reached=ro["reached"],
               act_probs=ro["act_probs"].astype(np.float32), value=value.astype(np.float32),
               start_nor=start_nor, start_R=start_R)
    np.savez_compressed(path, **out)
    for i, name in enumerate(names):
        sel = setting == i
        dist = maze.dist[starts[sel]]
        log(f"  {name:8s} n={sel.sum()}  start dist {dist.mean():4.1f} (min {dist.min()}, max {dist.max()})  "
            f"reached {ro['reached'][sel].mean():.2f}  mean length {ro['length'][sel].mean():6.1f}")
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return out


def load_testset(path=TEST_PATH):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# scoring a model against the exact targets
# ---------------------------------------------------------------------------

def stratified_rows(ts, per_setting, seed=0):
    """Row indices with at most `per_setting` rows from every setting (all rows if per_setting is None)."""
    if per_setting is None:
        return np.arange(len(ts["setting"]))
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(ts["setting"] == i))[:per_setting]
                           for i in range(len(ts["setting_names"]))])


def _log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _kl(p, logq):
    return np.where(p > 0, p * (np.log(np.maximum(p, 1e-30)) - logq), 0.0).sum(-1)


def score(params, fwd, tok: Tokenizer, ts, rows=None, chunk=200) -> dict:
    """Exact-target metrics per setting, in nats:
        act_kl    KL(true || model) of the action distribution at state slots, row's own MODE
        value_kl  KL(h || V) of the value head at state slots, prefix read in NOR mode
        start_kl  KL(true || model) of the start-cell distribution at the MODE slot
        dyn_nll   -log p(recorded next cell) at action slots (0 is perfect; the dynamics are deterministic)
    Returns flat keys "<metric>/<setting>" plus "per_setting" = {setting: {metric: value}}.
    fwd is model.make_forward(model, tok)."""
    import jax.numpy as jnp
    rows = np.arange(len(ts["setting"])) if rows is None else np.asarray(rows)
    T = tok.T
    x = ts["tokens"][rows].astype(np.int32)
    x_nor = x.copy()
    x_nor[:, 0] = tok.mode(None)
    c0, c1 = tok.OUT_CELL0, tok.OUT_END
    pi, start, dyn, v = [], [], [], []
    for i in range(0, len(rows), chunk):
        o = fwd(params, jnp.asarray(x[i:i + chunk]))
        pi.append(np.asarray(o["pi_logits"][:, :T]))
        start.append(np.asarray(o["next"][:, 0, c0:c1]))
        dyn.append(np.asarray(o["dyn_logits"]))
        v.append(np.asarray(fwd(params, jnp.asarray(x_nor[i:i + chunk]))["v_logits"]))
    pi, start, dyn, v = (np.concatenate(z) for z in (pi, start, dyn, v))
    L = ts["length"][rows].astype(np.int64)
    valid_a = np.arange(T)[None] < L[:, None]
    valid_s = np.arange(T + 1)[None] <= L[:, None]
    act_kl = _kl(ts["act_probs"][rows].astype(np.float64), _log_softmax(pi))                    # [n, T]
    value_kl = _kl(ts["value"][rows].astype(np.float64), _log_softmax(v))                       # [n, T+1]
    mb = ts["mode_bin"][rows]
    start_true = np.where((mb < 0)[:, None], ts["start_nor"][None], ts["start_R"][np.maximum(mb, 0)])
    start_kl = _kl(start_true, _log_softmax(start))                                             # [n]
    nxt = ts["positions"][rows][:, 1:].astype(np.int64)
    dyn_nll = -np.take_along_axis(_log_softmax(dyn), nxt[..., None], -1)[..., 0]                 # [n, T]
    out, per = {}, {}
    names = [str(s) for s in ts["setting_names"]]
    for i, name in enumerate(names):
        sel = ts["setting"][rows] == i
        if not sel.any():
            continue
        per[name] = dict(act_kl=float(act_kl[sel][valid_a[sel]].mean()),
                         value_kl=float(value_kl[sel][valid_s[sel]].mean()),
                         start_kl=float(start_kl[sel].mean()),
                         dyn_nll=float(dyn_nll[sel][valid_a[sel]].mean()))
        for k, val in per[name].items():
            out[f"{k}/{name}"] = val
    out["per_setting"] = per
    return out
