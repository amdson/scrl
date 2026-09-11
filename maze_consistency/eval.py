"""Metrics against the DP ground truth, and rollouts with the model as policy."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .env import (Maze, rollout_numpy, random_walk_policy, table_policy, dp_state, step_env,
                  FLAG_UNKNOWN, FLAG_OPEN, FLAG_LOCKED, N_ACTIONS)
from .dp import GroundTruth
from .tokens import Tokenizer, r_distribution


def _softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def _log_softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis, keepdims=True))


def rtg_to_total(logV, t0=0):
    """Convert value log-probs over return-to-go bins to total-return bins.
    logV [..., T', K] indexed by step (first step = t0): total bin k at step t = rtg bin k + t."""
    K = logV.shape[-1]
    T1 = logV.shape[-2]
    t = (np.arange(T1) + t0)[:, None]
    k = np.arange(K)[None, :]
    src = np.where(k == 0, 0, k + t)                                # [T', K]; bin 0 = FAIL stays put
    ok = src <= K - 1
    src = np.minimum(src, K - 1)
    out = np.take_along_axis(logV, np.broadcast_to(src, logV.shape), -1)
    return np.where(ok, out, -60.0)


def forward_np(fwd, params, tokens, chunk=512):
    outs = []
    for i in range(0, tokens.shape[0], chunk):
        o = fwd(params, jnp.asarray(tokens[i:i + chunk]))
        outs.append({k: np.asarray(v) for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}


class EvalSet:
    """Fixed random-walk eval episodes + DP-optimal prefixes, with ground truth attached."""

    def __init__(self, maze: Maze, gt: GroundTruth, tok: Tokenizer, n=2000, n_opt=512, seed=12345):
        self.maze, self.gt, self.tok = maze, gt, tok
        T, K = maze.T, maze.K
        rng = np.random.default_rng(seed)
        self.r_probs = r_distribution(maze)
        d = rollout_numpy(maze, random_walk_policy, n, rng)
        self.data = d
        self.s = dp_state(maze, d["positions"], d["flags"])                     # [n, T+1]
        self.t_idx = np.broadcast_to(np.arange(T + 1)[None], self.s.shape)
        self.valid_state = np.arange(T + 1)[None] <= d["length"][:, None]
        self.valid_act = np.arange(T)[None] < d["length"][:, None]
        self.rs_bin = rng.choice(K, size=n, p=self.r_probs)
        body = tok.encode_body(d["positions"], d["flags"], d["actions"], d["length"])
        self.tok_nor = tok.with_mode(body, None)
        self.tok_Rs = tok.with_mode(body, self.rs_bin)
        self.h_true = gt.h[self.t_idx, self.s]                                   # [n, T+1, K]
        self.er_true = (self.h_true * maze.R_values).sum(-1)
        # pi_R* at the sampled R, for t < T
        self.piR_true = gt.piR_star[self.t_idx[:, :T], self.s[:, :T], self.rs_bin[:, None]]   # [n,T,4]
        self.piR_defined = np.isfinite(self.piR_true).all(-1) & self.valid_act
        # (A) residual support: h[t,s,R]>0 and h[t+1,s',R]>0
        h_rs = self.h_true[np.arange(n)[:, None], np.arange(T + 1)[None], self.rs_bin[:, None]]   # [n,T+1]
        self.cons_mask = (h_rs[:, :T] > 0) & (h_rs[:, 1:] > 0) & self.valid_act
        self.log_h_rs = np.log(np.maximum(h_rs, 1e-30))
        # DP-optimal prefixes (uniform over shortest paths) for opt_err
        kstar = int(maze.bin_of(maze.R_max))
        self.kstar = kstar
        o = rollout_numpy(maze, table_policy(gt.policy_piR(maze.R_max), maze), n_opt, rng)
        self.opt = o
        self.opt_s = dp_state(maze, o["positions"], o["flags"])
        self.opt_valid_act = np.arange(T)[None] < o["length"][:, None]
        self.opt_piR_true = gt.piR_star[np.arange(T)[None], self.opt_s[:, :T], kstar]      # [n_opt,T,4]
        self.opt_defined = np.isfinite(self.opt_piR_true).all(-1) & self.opt_valid_act
        self.tok_opt_Rstar = tok.with_mode(tok.encode_body(o["positions"], o["flags"], o["actions"], o["length"]), kstar)
        # FQI truth
        self.V_opt_true = gt.V_opt[self.t_idx, self.s]


def _mean_over_t(x, mask):
    """x, mask [n, T'] -> per-t mean over rows where mask; nan where no rows."""
    m = mask.astype(np.float64)
    num = np.where(mask, x, 0.0).sum(0)
    den = m.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def kl(p, logq, eps=1e-12):
    """KL(p || q) with q given as log-probs; p may have zeros."""
    return np.where(p > 0, p * (np.log(np.maximum(p, eps)) - logq), 0.0).sum(-1)


def evaluate_model(params, fwd, es: EvalSet, rollouts=True, rollout_n=512, betas=(1.0, 3.0, 10.0),
                   seed=0, fqi=False, detail=False, rtg=False) -> dict:
    maze, gt, tok = es.maze, es.gt, es.tok
    T, K = maze.T, maze.K
    n = es.tok_nor.shape[0]
    out_nor = forward_np(fwd, params, es.tok_nor)
    out_rs = forward_np(fwd, params, es.tok_Rs)
    logV = _log_softmax(out_nor["value"])                       # [n,T+1,K]
    if rtg:
        logV = rtg_to_total(logV)
    V = np.exp(logV)
    logpi = _log_softmax(out_nor["action"])
    logpiRs = _log_softmax(out_rs["action"])
    m = {}
    tv = 0.5 * np.abs(V - es.h_true).sum(-1)
    m["value_err"] = _mean_over_t(tv, es.valid_state)
    er = (V * maze.R_values).sum(-1)
    m["er_err"] = _mean_over_t(np.abs(er - es.er_true), es.valid_state)
    # P(R = R*) log-ratio error: how well the rare optimal bin is tracked
    ks = es.kstar
    lr = np.abs(logV[..., ks] - np.log(np.maximum(es.h_true[..., ks], 1e-30)))
    m["logp_rstar_err"] = _mean_over_t(lr, es.valid_state & (es.h_true[..., ks] > 0))
    m["pi_kl_uniform"] = float(np.nanmean(_mean_over_t(kl(np.full((n, T, 4), 0.25), logpi[:, :T]), es.valid_act)))
    piR_kl = kl(np.nan_to_num(es.piR_true), logpiRs[:, :T])
    m["piR_err"] = _mean_over_t(piR_kl, es.piR_defined)
    # (A) residual at data actions, R ~ r(R)
    act = es.data["actions"].astype(np.int64)
    lp_a = np.take_along_axis(logpi[:, :T], act[..., None], -1)[..., 0]
    lpR_a = np.take_along_axis(logpiRs[:, :T], act[..., None], -1)[..., 0]
    lV_rs = np.take_along_axis(logV, es.rs_bin[:, None, None].repeat(T + 1, 1), -1)[..., 0]     # [n,T+1]
    resid = np.abs(lpR_a + lV_rs[:, :T] - lp_a - lV_rs[:, 1:])
    m["consistency"] = _mean_over_t(resid, es.cons_mask)
    # opt_err on DP-optimal prefixes at R = R*
    out_opt = forward_np(fwd, params, es.tok_opt_Rstar)
    logpi_opt = _log_softmax(out_opt["action"])[:, :T]
    opt_kl = kl(np.nan_to_num(es.opt_piR_true), logpi_opt)
    m["opt_err_t"] = _mean_over_t(opt_kl, es.opt_defined)
    m["opt_err"] = float(opt_kl[es.opt_defined].mean())
    if fqi:
        qmax = out_nor["q"].max(-1)
        m["q_err"] = _mean_over_t(np.abs(qmax - es.V_opt_true), es.valid_state)
        m["overest"] = _mean_over_t(qmax - es.V_opt_true, es.valid_state)
    m["max_value_err"] = float(np.nanmax(m["value_err"]))
    m["mean_value_err"] = float(np.nanmean(m["value_err"]))
    m["mean_consistency"] = float(np.nanmean(m["consistency"]))
    m["mean_piR_err"] = float(np.nanmean(m["piR_err"]))
    if rollouts:
        rng = np.random.default_rng(seed)
        pols = [("piR_sample", dict(policy="piR")), ("piR_greedy", dict(policy="piR", greedy=True)),
                ("posterior", dict(policy="posterior")), ("posterior_greedy", dict(policy="posterior", greedy=True))]
        pols += [(f"ev_b{b:g}", dict(policy="ev", beta=b)) for b in betas]
        if fqi:
            pols += [("fqi_greedy", dict(policy="fqi", greedy=True))]
        m["rollouts"] = {}
        for name, kw in pols:
            ro = rollout_model(params, fwd, tok, maze, N=rollout_n, rng=rng, rtg=rtg, **kw)
            m["rollouts"][name] = summarize_rollout(maze, ro)
    if detail:
        m["_detail"] = cell_maps(maze, gt, es, V, logpi_opt)
    return m


def summarize_rollout(maze: Maze, ro: dict) -> dict:
    solved = ro["returns"] > -(maze.T + 1)
    return dict(solve_rate=float(solved.mean()),
                mean_steps=float(ro["length"][solved].mean()) if solved.any() else float("nan"),
                mean_return=float(ro["returns"].mean()),
                optimal_rate=float((ro["returns"] == maze.R_max).mean()),
                door_choice=float(ro["door_attempted"].mean()) if maze.has_door else None)


def cell_maps(maze: Maze, gt: GroundTruth, es: EvalSet, V, logpi_opt) -> dict:
    """Per-cell aggregates for visualisation: model vs true P(R=R*|t,s) and pi_R(R*) arrows."""
    T, n_cells = maze.T, maze.n_cells
    ks = es.kstar
    pos = es.data["positions"]
    out = {"t_list": [], "p_rstar_model": [], "p_rstar_true": []}
    for t in range(0, min(T, maze.d_star + 1)):
        vm = np.full(n_cells, np.nan)
        vt = np.full(n_cells, np.nan)
        msk = es.valid_state[:, t]
        for c in np.unique(pos[msk, t]):
            rows = msk & (pos[:, t] == c)
            vm[c] = V[rows, t, ks].mean()
            vt[c] = es.h_true[rows, t, ks].mean()
        out["t_list"].append(t)
        out["p_rstar_model"].append(vm)
        out["p_rstar_true"].append(vt)
    # pi_R(R*) per cell (model) on DP-optimal prefixes, and the DP reference
    arr_m = np.full((n_cells, 4), np.nan)
    arr_t = np.full((n_cells, 4), np.nan)
    opos = es.opt["positions"][:, :T]
    pm = np.exp(logpi_opt)
    for c in np.unique(opos[es.opt_defined]):
        rows = es.opt_defined & (opos == c)
        arr_m[c] = pm[rows].mean(0)
        arr_t[c] = np.nan_to_num(es.opt_piR_true[rows]).mean(0)
    out["piR_star_model"] = arr_m
    out["piR_star_true"] = arr_t
    return out


# ---------------------------------------------------------------------------
# rollouts with the model as policy
# ---------------------------------------------------------------------------

def rollout_model(params, fwd, tok: Tokenizer, maze: Maze, policy="piR", N=512, rng=None, R=None,
                  beta=1.0, greedy=False, eps=1e-9, rtg=False) -> dict:
    """policy in {piR, posterior, ev, fqi, random}. R defaults to R_max (= -d*).
    posterior: p(a) ∝ pi(a|s) V_{t+1}(R | s, a);  ev: p(a) ∝ pi(a|s) exp(beta E_V[R | s, a]).
    V(R|s,a) is computed from explicit child sequences (expected over the door flag)."""
    rng = rng or np.random.default_rng(0)
    T, K = maze.T, maze.K
    R = maze.R_max if R is None else R
    kR = int(maze.bin_of(R))
    tokens = np.full((N, tok.L), tok.PAD, dtype=np.int32)
    tokens[:, 0] = tok.mode_token(kR) if policy == "piR" else tok.NOR
    if tok.use_grid:
        tokens[:, 1:1 + tok.n_grid] = tok.grid_tokens[None]
    pos = np.full(N, maze.start, dtype=np.int32)
    flag = np.zeros(N, dtype=np.int8)
    locked = rng.random(N) < maze.p_locked
    positions = np.zeros((N, T + 1), dtype=np.int32)
    flags = np.zeros((N, T + 1), dtype=np.int8)
    actions = np.zeros((N, T), dtype=np.int8)
    length = np.full(N, T, dtype=np.int32)
    returns = np.full(N, -(T + 1), dtype=np.int32)
    door_attempted = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    positions[:, 0] = pos
    Rv = maze.R_values
    for t in range(T):
        if policy == "random":
            probs = np.full((N, 4), 0.25)
        else:
            out = fwd(params, jnp.asarray(tokens))
            if policy == "piR":
                probs = _softmax(np.asarray(out["action"][:, t]))
            elif policy == "fqi":
                q = np.asarray(out["q"][:, t])
                probs = np.eye(4)[q.argmax(-1)]
            else:
                logpi = _log_softmax(np.asarray(out["action"][:, t]))
                childV = np.zeros((N, 4, K))
                branches = [(maze.next_open, 0)] + ([(maze.next_locked, 1)] if maze.has_door else [])
                for a in range(4):
                    for nxt_tab, br in branches:
                        npos = nxt_tab[pos, a]
                        nflag = flag.copy()
                        w = np.ones(N)
                        if maze.has_door:
                            attempted = (maze.next_open[pos, a] == maze.door) & (pos != maze.door)
                            unk = flag == FLAG_UNKNOWN
                            if br == 0:   # open-table branch
                                nflag = np.where(npos == maze.door, FLAG_OPEN, flag)
                                w = np.where(flag == FLAG_LOCKED, 0.0, np.where(unk & attempted, 1 - maze.p_locked, 1.0))
                            else:         # locked-table branch
                                nflag = np.where(attempted, FLAG_LOCKED, flag)
                                w = np.where(flag == FLAG_LOCKED, 1.0, np.where(unk & attempted, maze.p_locked, 0.0))
                        ct = tokens.copy()
                        ct[:, tok.sidx[t] + 1] = tok.ACT0 + a
                        ct[:, tok.sidx[t] + 2] = tok.pos_token(npos, nflag)
                        lv = _log_softmax(np.asarray(fwd(params, jnp.asarray(ct))["value"][:, t + 1]))
                        if rtg:
                            lv = rtg_to_total(lv[:, None, :], t0=t + 1)[:, 0]
                        childV[:, a] += w[:, None] * np.exp(lv)
                if policy == "posterior":
                    wgt = np.exp(logpi) * childV[:, :, kR]
                    z = wgt.sum(-1, keepdims=True)
                    probs = np.where(z > eps, wgt / np.maximum(z, eps), np.exp(logpi))
                elif policy == "ev":
                    ev = (childV * Rv).sum(-1)
                    logits = logpi + beta * (ev - ev.max(-1, keepdims=True))
                    probs = _softmax(logits)
                else:
                    raise ValueError(policy)
        if greedy:
            a = probs.argmax(-1)
        else:
            cum = np.cumsum(probs, -1)
            a = np.minimum((rng.random(N)[:, None] > cum).sum(-1), 3)
        a = a.astype(np.int8)
        nxt, nflag, attempted = step_env(maze, pos, flag, locked, a)
        pos = np.where(alive, nxt, pos)
        flag = np.where(alive, nflag, flag)
        door_attempted |= alive & attempted
        actions[:, t] = a
        positions[:, t + 1] = pos
        flags[:, t + 1] = flag
        # write the step into the token buffer for still-alive episodes
        tokens[alive, tok.sidx[t] + 1] = tok.ACT0 + a[alive]
        tokens[alive, tok.sidx[t] + 2] = tok.pos_token(pos[alive], flag[alive])
        reached = alive & (pos == maze.goal)
        length[reached] = t + 1
        returns[reached] = -(t + 1)
        alive &= ~reached
        if not alive.any():
            break
    return dict(positions=positions, flags=flags, actions=actions, length=length, returns=returns,
                locked=locked, door_attempted=door_attempted)


def dp_reference_rollouts(maze: Maze, gt: GroundTruth, N=2000, seed=0, betas=(1.0, 3.0, 10.0)) -> dict:
    """What each decode rule does with *perfect* values (the DP tables)."""
    rng = np.random.default_rng(seed)
    tabs = {"random": np.full_like(gt.Q_rw, 0.25), "piR_star": gt.policy_piR(maze.R_max),
            "posterior": gt.policy_posterior_tilt(maze.R_max), "greedy_opt": gt.policy_greedy_opt()}
    for b in betas:
        tabs[f"ev_b{b:g}"] = gt.policy_ev_tilt(b)
    return {k: summarize_rollout(maze, rollout_numpy(maze, table_policy(v, maze), N, rng)) for k, v in tabs.items()}
