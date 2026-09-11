"""Losses. All are computed on-sequence: every data step t contributes, and the (B)/(A) child
terms use the data's own next step as an unbiased single sample of the 4-child expectation
(the behaviour policy q is the random walk, so pi(a)/q(a) re-weights exactly).

    L_pi    = CE(pi(.|s_t), a_{t+1})                              NOR mode
    L_piR   = CE(pi_R(.|s_t, R_ep), a_{t+1})                      R = episode return
    L_MC    = CE(V_t, R_ep)                                        all t <= L
    L_term  = CE(V_L, R_ep)                                        t = L only (exact base case)
    L_TD    = CE(V_t, sg[ (pi(a|s_t)/q(a|s_t)) V_{t+1}(.|s_{t+1}) ])          (B), t < L
    L_A     = - sg[ (pi(a|s)/q(a|s)) V_{t+1}(R|s')/V_t(R|s) ] log pi_R(a|s,R)   (A), R ~ r(R)

The importance weights are exactly <= 1/q = 4 when (B) holds; they are clamped at w_max.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

sg = jax.lax.stop_gradient


@dataclass
class LossConfig:
    use_pi: bool = True
    use_piR: bool = True
    use_mc: bool = False
    use_td: bool = False
    use_term: bool = False
    use_a: bool = False
    w_pi: float = 1.0
    w_piR: float = 1.0
    w_mc: float = 1.0
    w_td: float = 1.0
    w_term: float = 1.0
    w_a: float = 1.0
    a_warmup: int = 0          # steps of TD before L_A is switched on
    w_max: float = 8.0
    td_backup: str = "pi"      # "pi" (identity B) or "piR" (closure 3: pi_R(R*) inside the backup for bin R*)
    distill: bool = False      # closure 2: L_pi -> CE(pi, sg[teacher pi_R(.|s,R*)])
    rtg: bool = False          # value bins: [FAIL, rtg=-T..0] (t-independent terminals) instead of total return
    use_children: bool = False # plan-faithful (A)/(B) on explicit 4-child prefixes (analytic terminal)

    @staticmethod
    def named(name: str, **kw) -> "LossConfig":
        base = dict(
            MC=dict(use_mc=True),
            TD=dict(use_td=True, use_term=True),
            TDA=dict(use_td=True, use_term=True, use_a=True),
            TDMC=dict(use_td=True, use_term=True, use_mc=True),
        )[name]
        base.update(kw)
        return LossConfig(**base)


def _masked_mean(x, mask):
    mask = mask.astype(x.dtype)
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def _gather_last(x, idx):
    """x [B, T', C], idx [B] -> x[b, :, idx[b]] : [B, T']"""
    return jnp.take_along_axis(x, idx[:, None, None], axis=-1)[..., 0]


def _bin_at_t(ret_bin, T1, K, rtg):
    """Per-step bin index [B, T1] of a total-return bin. rtg scheme: bin 0 = FAIL (constant),
    bins 1..K-1 = return-to-go -T..0, so total bin b>=1 at step t is b + t."""
    if not rtg:
        return jnp.broadcast_to(ret_bin[:, None], (ret_bin.shape[0], T1))
    b = ret_bin[:, None] + jnp.arange(T1)[None, :]
    return jnp.where(ret_bin[:, None] == 0, 0, jnp.clip(b, 0, K - 1))


def _gather_t(x, idx_t):
    """x [B, T', C], idx_t [B, T'] -> x[b, t, idx_t[b, t]]"""
    return jnp.take_along_axis(x, idx_t[..., None], axis=-1)[..., 0]


def _shift_child(V):
    """rtg scheme: FAIL bin stays; for k>=1 parent bin k draws on child bin k+1 (rtg_t = rtg_{t+1} - 1)."""
    return jnp.concatenate([V[..., :1], V[..., 2:], jnp.zeros_like(V[..., :1])], -1)


def compute_losses(heads, batch, cfg: LossConfig, step, teacher_heads=None):
    """heads: dict with 'nor', 'R', 'Rs' (and optionally 'Rstar'), each {'action':[B,T+1,4], 'value':[B,T+1,K]}.
    batch: act [B,T], valid_act [B,T], valid_state [B,T+1], term_mask [B,T+1], ret_bin [B], rs_bin [B],
           logq [B,T], rstar_bin scalar.
    """
    act = batch["act"]
    va, vs, tm = batch["valid_act"], batch["valid_state"], batch["term_mask"]
    B, T = act.shape

    logpi = jax.nn.log_softmax(heads["nor"]["action"], -1)          # [B,T+1,4]
    logV = jax.nn.log_softmax(heads["nor"]["value"], -1)            # [B,T+1,K]
    logpiR = jax.nn.log_softmax(heads["R"]["action"], -1)
    logpiRs = jax.nn.log_softmax(heads["Rs"]["action"], -1)

    logpi_a = jnp.take_along_axis(logpi[:, :T], act[..., None], -1)[..., 0]     # [B,T]
    logpiR_a = jnp.take_along_axis(logpiR[:, :T], act[..., None], -1)[..., 0]
    logpiRs_a = jnp.take_along_axis(logpiRs[:, :T], act[..., None], -1)[..., 0]
    K = logV.shape[-1]
    logV_ret = _gather_t(logV, _bin_at_t(batch["ret_bin"], T + 1, K, cfg.rtg))   # [B,T+1]

    losses = {}
    losses["pi"] = _masked_mean(-logpi_a, va)
    losses["piR"] = _masked_mean(-logpiR_a, va)
    losses["mc"] = _masked_mean(-logV_ret, vs)
    losses["term"] = _masked_mean(-logV_ret, tm)

    # ---- (B): on-sequence TD with importance weight pi/q --------------------------------
    log_w = logpi_a - batch["logq"]                                              # [B,T]
    w = jnp.clip(jnp.exp(sg(log_w)), 0.0, cfg.w_max)
    target = sg(jnp.exp(logV[:, 1:]))                                            # V_{t+1}  [B,T,K]
    if cfg.rtg:
        target = _shift_child(target)
    if cfg.td_backup == "piR" and "Rstar" in heads:
        # closure 3: for bin R*, weight the backup by pi_R(a|s,R*) instead of pi(a|s)
        logpiRstar = jax.nn.log_softmax(heads["Rstar"]["action"], -1)
        lp_star = jnp.take_along_axis(logpiRstar[:, :T], act[..., None], -1)[..., 0]
        w_star = jnp.clip(jnp.exp(sg(lp_star - batch["logq"])), 0.0, cfg.w_max)
        kstar = batch["rstar_bin"]
        onehot = jax.nn.one_hot(kstar, logV.shape[-1])[None, None]
        wfull = w[..., None] * (1 - onehot) + w_star[..., None] * onehot
        td_target = wfull * target
    else:
        td_target = w[..., None] * target
    losses["td"] = _masked_mean(-(td_target * logV[:, :T]).sum(-1), va)

    # ---- (A): importance-weighted CE for pi_R at R ~ r(R) ---------------------------------
    rs_t = _bin_at_t(batch["rs_bin"], T + 1, K, cfg.rtg)
    logV_rs = _gather_t(logV, rs_t)                                              # [B,T+1]
    log_wa = log_w + logV_rs[:, 1:] - logV_rs[:, :T]
    wa = jnp.clip(jnp.exp(sg(log_wa)), 0.0, cfg.w_max)
    va_a = va & ((batch["rs_bin"][:, None] == 0) | ((batch["rs_bin"][:, None] + jnp.arange(1, T + 1)[None]) <= K - 1)) if cfg.rtg else va
    losses["a"] = _masked_mean(-wa * logpiRs_a, va_a)
    losses["a_wmean"] = _masked_mean(wa, va_a)        # diagnostic: ~1 when (B) holds

    # ---- explicit children (plan §4): exact (B) and (A) on a subsample of prefixes ------------
    if cfg.use_children and "child" in heads:
        cb = batch["child"]
        ci, ct = cb["idx"], cb["t"]                                              # [Bc]
        Bc, A, F = cb["w"].shape
        # child value at step t+1 of each child sequence
        cv = heads["child"]["value"]                                              # [Bc*A*F, T+1, K]
        cv = jnp.take_along_axis(cv, jnp.repeat(ct + 1, A * F)[:, None, None], axis=1)[:, 0]   # [Bc*A*F, K]
        Vc = sg(jax.nn.softmax(cv, -1)).reshape(Bc, A, F, K)
        # analytic terminal: goal -> onehot(-(t+1)) [or rtg 0]; timeout -> onehot(fail)
        term_bin = jnp.full((Bc,), K - 1) if cfg.rtg else jnp.clip(K - 2 - ct, 0, K - 1)   # total: bin(-(t+1)) = T+1-(t+1)
        fail_bin = jnp.zeros_like(ct)                                             # FAIL bin is 0 in both schemes
        Vc = jnp.where(cb["goal"][..., None], jax.nn.one_hot(term_bin, K)[:, None, None, :], Vc)
        Vc = jnp.where(cb["timeout"][..., None], jax.nn.one_hot(fail_bin, K)[:, None, None, :], Vc)
        Vc = (cb["w"][..., None] * Vc).sum(2)                                     # [Bc, A, K]  = V_{t+1}(.|s,a)
        if cfg.rtg:
            Vc = _shift_child(Vc)
        pi_c = sg(jnp.exp(logpi[ci, ct]))                                         # [Bc, A]
        tgt = (pi_c[..., None] * Vc).sum(1)                                       # [Bc, K]  (B) rhs, normalised
        logV_par = logV[ci, ct]                                                   # [Bc, K]
        losses["td_child"] = -(tgt * logV_par).sum(-1).mean()
        # (A): target over actions for R = rs_bin[ci] at time t
        rs_c = _bin_at_t(batch["rs_bin"][ci], 1, K, cfg.rtg)[:, 0] if not cfg.rtg else jnp.where(batch["rs_bin"][ci] == 0, 0, jnp.clip(batch["rs_bin"][ci] + ct, 0, K - 1))
        num = pi_c * jnp.take_along_axis(Vc, rs_c[:, None, None].repeat(A, 1), -1)[..., 0]   # [Bc, A]
        Z = num.sum(-1, keepdims=True)
        tgt_a = num / jnp.maximum(Z, 1e-30)
        defined = (Z[:, 0] > 1e-8)
        logpiRs_c = logpiRs[ci, ct]                                               # [Bc, A]
        losses["a_child"] = _masked_mean(-(tgt_a * logpiRs_c).sum(-1), defined)
        losses["a_child_frac"] = defined.astype(jnp.float32).mean()

    # ---- closure 2: distillation pi <- pi_R(R*) --------------------------------------------
    if cfg.distill and teacher_heads is not None:
        t_logp = jax.nn.log_softmax(teacher_heads["Rstar"]["action"], -1)[:, :T]
        t_p = sg(jnp.exp(t_logp))
        losses["distill"] = _masked_mean(-(t_p * logpi[:, :T]).sum(-1), va)

    a_on = (step >= cfg.a_warmup).astype(jnp.float32)
    total = 0.0
    if cfg.use_pi and not cfg.distill:
        total = total + cfg.w_pi * losses["pi"]
    if cfg.distill and "distill" in losses:
        total = total + cfg.w_pi * losses["distill"]
    if cfg.use_piR:
        total = total + cfg.w_piR * losses["piR"]
    if cfg.use_mc:
        total = total + cfg.w_mc * losses["mc"]
    if cfg.use_td:
        total = total + cfg.w_td * losses["td"]
    if cfg.use_term:
        total = total + cfg.w_term * losses["term"]
    if cfg.use_a:
        total = total + cfg.w_a * a_on * losses["a"]
    if cfg.use_children and "td_child" in losses:
        if cfg.use_td:
            total = total + cfg.w_td * losses["td_child"]
        if cfg.use_a:
            total = total + cfg.w_a * a_on * losses["a_child"]
    losses["total"] = total
    return total, losses


def fqi_loss(q_heads, q_target_heads, batch, maze_T, goal_reached_next):
    """Neural FQI: time-indexed scalar Q with max backup and a target network.
    q_heads['q'] [B,T+1,4] (online), q_target_heads['q'] (target params).
    goal_reached_next [B,T] bool: pos_{t+1} == goal."""
    act, va = batch["act"], batch["valid_act"]
    B, T = act.shape
    q = q_heads["q"][:, :T]
    q_a = jnp.take_along_axis(q, act[..., None], -1)[..., 0]
    q_next = sg(q_target_heads["q"][:, 1:].max(-1))                               # [B,T]
    t_next = jnp.arange(1, T + 1)[None, :]
    done = goal_reached_next | (t_next >= maze_T)
    reward = -1.0 - ((t_next >= maze_T) & ~goal_reached_next).astype(jnp.float32)  # -1 more if failed
    y = reward + jnp.where(done, 0.0, q_next)
    err = q_a - y
    huber = jnp.where(jnp.abs(err) < 1.0, 0.5 * err ** 2, jnp.abs(err) - 0.5)
    return _masked_mean(huber, va), {"fqi": _masked_mean(huber, va), "q_mean": _masked_mean(q_a, va)}
