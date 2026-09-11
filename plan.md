# Minimal test: reward-conditioned self-consistency on a fixed maze (JAX)

## 0. What is being tested

A single causal transformer over `[R-or-NOR] [grid] (a_1, pos_1) (a_2, pos_2) ...` with three readouts:

- `pi(a_t | prefix)`            — next-action head, NOR mode
- `V_t(R | prefix)`             — categorical reward head at every step, NOR mode
- `pi_R(a_t | prefix, R)`       — next-action head, R mode (R token as prefix)

Consistency identities (fixed token order, R floats):

```
(A)  pi_R(a|s,R) * V_{t-1}(R|s)  =  pi(a|s) * V_t(R|s,a)          for all t, s, a, R
(B)  V_{t-1}(R|s)               =  sum_a pi(a|s) V_t(R|s,a)        (marginal of A)
```

Hypotheses, in the order they should be killed:

| ID | Claim | Kill criterion |
|----|-------|----------------|
| H1 | TD via (B) gives lower early-t value error than MC regression on random-walk data | TD curve not below MC curve for t < T/2 at any data size |
| H2 | (A) recovers the optimal policy (uniform over shortest paths) at R = -d*, a value never observed in data | KL(pi_R* ‖ pi_R) at R=-d* not < 0.1 nats averaged over states |
| H3 | Posterior tilt gambles, expected-value tilt doesn't, from the same heads | Both tilts pick the same route on the locked-door maze |
| H4 | Value error grows linearly in horizon and does not diverge; FQI on the same data does worse | Error curve superlinear, or FQI matches it |
| H5 | Each loop closure (off-policy q, pi <- pi_R, pi_R in backup) degrades H4 in the predicted order | No ordering visible |

Everything below is sized so E1–E4 run on one GPU (or a laptop CPU at reduced N) in under a day total.

---

## 1. Environment

**Maze.** Fixed layout, 9x9 open cells inside a wall border (so 11x11 grid tokens). Generated once by recursive backtracker, seed fixed. Start and goal fixed, shortest distance d* in [20, 30]. Store as `grid: int8[11,11]` with values {WALL, OPEN, START, GOAL}.

**Dynamics.** Actions {U, D, L, R}. Deterministic; moving into a wall = stay. Horizon T = 60 (≈ 2–3 d*). Episode ends at goal or T.

**Reward.** -1 per step. Episode return R in [-T, -d*]; not-reached episodes get R = -T - 1 (own bin). Discretize R to K = T + 2 categorical bins directly (one bin per integer). No coarser binning: it costs nothing here and keeps (B) exact.

**Locked-door variant (for H3).** Same maze plus a one-cell door on a shortcut. At episode start the door is locked with p = 0.7; locked/unlocked is revealed only in the pos token on arrival (add a 5th grid value DOOR_LOCKED visible in the pos-embedding, not in the prompt). Route via door: 10 steps if open, else 30 (dead-end detour). Route around: 12. Choose the layout so these are the only two sensible routes.

**Ground truth (numpy, once).**
- `h[t, s, r]` = P_randomwalk(return-to-go = r | state s at time t), by backward induction over t. This is `V*_t`.
- `pi_R*[t, s, r, a] = pi(a) * h[t+1, s', r+1] / h[t, s, r]`, the Doob h-transform. At r = -d* from the start state this is uniform over shortest paths.
- `Q*[t, s, a] = E[R | s, a]` under random walk, for the expected-value tilt reference.
- For the door variant, the state includes the door flag once observed; do DP over (pos, door_flag ∈ {unknown, open, locked}).

---

## 2. Data

Random-walk episodes only. `pi_data(a|s) = 1/4`.

- N ∈ {2k, 10k, 50k} episodes (sweep for H1).
- Store `actions int8[N,T]`, `positions int16[N,T]` (flattened cell index), `returns int16[N]`, `length int16[N]`.
- Tokenization per episode: `[MODE] [grid 121 tokens] [(a_1,pos_1) ... (a_L,pos_L)] [EOS]`. Action and position are separate tokens (so sequence length ≈ 1 + 121 + 2T + 1 = 243; fine).
- MODE token is `R_k` (one of K) or `NOR`. Each data episode appears in both modes.

No model rollouts are needed for the acyclic configuration: `q` (the prefix distribution for the consistency losses) is the random walk itself, which is the environment, not the model. Generate extra random-walk prefixes cheaply in numpy. Model rollouts only enter in E4/E5 loop closures.

---

## 3. Model (flax.linen, single module)

```
d_model=128, n_layers=4, n_heads=4, seq_len=256, vocab ≈ 4 grid + 5 door + 4 actions + 121 pos + K modes + NOR + EOS
```

- Causal attention. Learned absolute positions. Token-type embedding (grid / action / pos / mode) added.
- Heads, all from the same residual stream:
  - `action_logits[4]` at every position where the next token is an action (i.e., at pos tokens and at the last grid token).
  - `value_logits[K]` at every pos token (and at the last grid token for t=0). Read only in NOR mode.
- `pi` = action head in NOR mode. `pi_R` = action head in R mode. `V_t` = value head in NOR mode at step t.

Param count ≈ 0.8M. One forward = both modes stacked in the batch dim.

Time-indexing is implicit (position in sequence), so `V_t` and `V_{t+1}` are different positions of the same NOR-mode pass — (B) needs one forward per batch, not two.

---

## 4. Losses

All on a batch of episodes/prefixes. `sg` = `jax.lax.stop_gradient`.

```
L_pi    = CE(action_head_NOR, a_t)                         # data, all t
L_piR   = CE(action_head_R,   a_t)                         # data, all t, R = episode return
L_MC    = CE(value_head_NOR at t, R_episode)               # data, all t   (baseline value target)
L_TD    = CE(value_head_NOR at t-1,
             sg[ sum_a pi(a|s_{t-1}) * V_t(R|s_{t-1},a) ]) # (B), all t, over prefixes
L_A     = KL( sg[ pi(a|s) V_t(R|s,a) / Z ]  ||  pi_R(a|s,R) )   # (A), over prefixes and R ~ r(R)
L_term  = CE(value_head_NOR at final step, R_episode)      # exact base case
```

Notes:
- `V_t(R|s,a)` for all 4 actions from one prefix requires the value at the 4 child states. Do it by constructing the 4 children explicitly (deterministic dynamics; door variant: expected over door flag using the model's own `V` at the two observed-flag children weighted by the empirical p). Batch: each prefix contributes 4 child sequences — this is the dominant cost; subsample to 1 random child per prefix per step for speed, unbiased for (B) since it's an expectation.
- `r(R)` for L_A: uniform over the K bins with 3x weight on the top 5 bins (near -d*). The identity holds for all R; signal is where V disagrees.
- Configurations:
  - `MC`      : L_pi + L_piR + L_MC
  - `TD`      : L_pi + L_piR + L_TD + L_term
  - `TD+A`    : TD + L_A
  - `TD+A(all)`: also apply L_A on random-walk prefixes not in the data (more coverage; same distribution)

Weights: 1.0 each; sweep only if a config fails.

---

## 5. Baselines

- **Tabular FQI** on the same transitions: scalar Q, max backup, 200 iterations; report Q error and overestimation `E[max_a Q_hat] - max_a Q*` per t. Cheap, exact, and the reference for "how bad is Q-learning here".
- **Neural FQI** with the same transformer trunk and a scalar Q head, time-indexed, max backup, target network updated every 500 steps. This is the apples-to-apples stability comparison for H4.
- **Plain DT / RCSL**: the `MC` config's `pi_R` conditioned on R = -d* (it has never seen that pair; expect garbage). This is the "what MLE alone gives" reference for H2.

---

## 6. Metrics

Computed against the DP ground truth, on all (t, s) reachable by the random walk, weighted by the random-walk occupancy `d_t(s)`:

```
value_err[t]   = E_{s~d_t} TV( V_t(.|s), h[t,s,.] )
piR_err[t]     = E_{s~d_t} E_{R~r} KL( pi_R*[t,s,R,.] || pi_R(.|s,R) )
opt_err        = KL( pi_R*[.,.,-d*,.] || pi_R(.|.,-d*) ) on shortest-path states
consistency[t] = E_{s,a,R} | log pi_R + log V_{t-1} - log pi - log V_t |     (A residual)
overest[t]     = E[max_a Q_hat] - max_a Q*                                   (FQI only)
solve_rate     = fraction of greedy/sampled rollouts reaching goal, and mean steps, for:
                   pi_R at R=-d* | posterior tilt pi*V_t(R_max) | EV tilt pi*exp(beta*E[R]) | FQI greedy
door_choice    = P(route via door) under each decode policy               (H3)
```

Rollouts for `solve_rate` use the real environment with the model as policy: `lax.scan` over T steps, batch 1024.

---

## 7. Experiments

**E1 — value error, MC vs TD (H1).** Configs `MC`, `TD`; N ∈ {2k, 10k, 50k}; 3 seeds. Plot `value_err[t]` vs t. Expect MC flat-and-high for small t (few positives), TD rising roughly linearly from the terminal.

**E2 — recovering the optimal policy (H2).** Configs `MC`, `TD`, `TD+A`; N = 10k. Report `opt_err` and `solve_rate` for `pi_R` at R = -d*. Expect `MC` ≈ random, `TD` unchanged (pi_R untouched by TD), `TD+A` near 0. This is the single experiment that tests the actual claim.

**E3 — posterior vs controller (H3).** Door variant, `TD+A`, N = 10k. Report `door_choice` for posterior tilt vs EV tilt (β sweep {1, 3, 10}). DP predicts: posterior → door (~1.0), EV → around (~1.0 for β large enough).

**E4 — stability (H4).** `TD+A` vs neural FQI, N = 10k, 5 seeds, 20k steps. Plot `value_err[t]` / Q error per t at checkpoints; plot max-over-t error vs training step. Expect TD+A monotone and bounded by ~linear-in-t; FQI with a seed-dependent overestimation drift.

**E5 — loop closures (H5).** Starting from `TD+A`, add one at a time:
1. `q` from `pi_R(R=-d*)` rollouts (off-policy state distribution; needs model rollouts inside training via `lax.scan`, every 100 steps, buffer 20k prefixes)
2. `pi <- pi_R` distillation each 2k steps (policy iteration)
3. `pi_R` instead of `pi` inside the (B) backup (soft Q-learning)

Report E4 curves for each. Predicted damage order: 3 > 2 > 1.

Run order: E1 → E2 (stop if H2 fails; nothing downstream matters) → E3 → E4 → E5.

---

## 8. JAX implementation notes

- **Libraries:** `jax`, `flax.linen`, `optax`, `numpy` for data/DP. No RL library needed.
- **Batching modes:** stack NOR and R sequences along batch; `mode` token differs only at position 0. One `jit`ted train step computes all losses.
- **Children for (B):** given a prefix batch `[B, L]`, build `[B, 4, L+2]` by appending `(a, pos')` for each action using the known dynamics (a lookup table `next_pos[pos, a]`); reshape to `[4B, L+2]`. With 1-child subsampling it's `[B, L+2]`.
- **Time indexing:** step t ↔ token index `1 + 121 + 2t`. Precompute an index array; gather value logits with `jnp.take_along_axis`.
- **Masking:** losses only at valid positions (before EOS); pass a `[B, L]` mask.
- **Rollouts:** `jax.lax.scan` over T with a KV-cache-free re-encode (L ≤ 256, B = 1024, 4 layers: fine). Only used for eval and E5.
- **Ground truth DP:** numpy, backward over t, arrays `h[T+1, 81, K]`; ~ms. Door variant: state = (pos, flag), 81*3.
- **Determinism:** `jax.random.PRNGKey(seed)`; data seeds separate from init seeds.
- **Compute:** 0.8M params × 256 tokens × B=256 → ~10 ms/step on a consumer GPU. 20k steps ≈ 4 min per run. E1–E5 total ≈ 60 runs ≈ 4–6 GPU-hours. CPU: reduce B to 64 and N to 10k; ~1 day.

Suggested layout:

```
maze_consistency/
  env.py          # maze gen, dynamics table, door variant, random-walk data
  dp.py           # h-transform ground truth, Q*, pi_R*
  tokens.py       # encoding, index maps, masks
  model.py        # flax transformer + heads
  losses.py       # L_pi, L_piR, L_MC, L_TD, L_A, L_term
  train.py        # configs MC / TD / TD+A / closures, optax, jit step
  fqi.py          # tabular + neural FQI baselines
  eval.py         # metrics vs DP, rollouts, plots
  run_E1.py ... run_E5.py
```

---

## 9. Things that will go wrong

- **Value head is all "not reached" early in the episode.** True for MC and correct for random walks with short T. Increase T or shrink the maze until `h[0, start, -d*]` ≥ 1e-3, else neither method has anything to learn.
- **(A) with stop-grad on the teacher but a bad V.** If TD hasn't converged, L_A distills a wrong posterior into pi_R. Warm up TD for 2k steps before enabling L_A; or anneal its weight.
- **Zero probability bins.** Use `log_softmax` throughout and clamp targets with ε = 1e-6 before the KL; never take `log` of a probability head output.
- **Children out of support in the door variant.** The unobserved-flag child is an expectation over the flag; if you accidentally evaluate V at an impossible (pos, flag) combination, the value is junk. Build children from the environment's true transition, not from the model.
- **FQI baseline "works fine".** With uniform random-walk coverage and a tiny maze, FQI often does work; that is not a failure of H4 but a sign the maze is too easy for the stability question. Reduce N to 2k and drop 30% of the actions from the data (biased behavior policy) to reintroduce coverage gaps; keep the same data for both methods.

---

## 10. Extension hooks (not part of the minimal test)

- Procedural mazes: prompt-conditioning; ground truth is per-maze DP, everything else unchanged.
- Sokoban/Boxoban: replace `env.py` and drop DP (use solver reachability for dead-end labels only).
- Replace `V_t(R|.)` with the partial-assignment energy from the earlier discussion: same head, R = validity bit, and (B) becomes the lattice marginalization constraint.