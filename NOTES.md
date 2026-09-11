# Notes on the maze self-consistency project

Written after a one-shot implementation of `plan.md` and a scaled-down (laptop CPU) run of E1–E5.
Sections: what was built and where it deviates from the plan · results · pitfalls · suggested tweaks.

## 1. What was built, and the deviations that matter

**Everything in the plan's layout exists** (`env / dp / tokens / model / losses / train / fqi / eval`,
plus `viz`, `experiments`, `run.py`, tests). Four deliberate deviations:

1. **On-sequence losses instead of explicit child sequences.** The plan builds `[B, 4, L+2]` child
   prefixes for (A) and (B). Both identities are expectations over `a ~ pi(.|s)`, and the training data
   *is* the random walk, so the data's own next step `(a_{t+1}, s_{t+1})` is an unbiased single-sample
   child with importance weight `pi(a|s)/q(a|s)` (q = 1/4). That gives:

   ```
   L_TD = CE( V_t , sg[ (pi(a|s_t)/q) * V_{t+1}(.|s_{t+1}) ] )                       all t < L, one forward
   L_A  = - sg[ (pi(a|s_t)/q) * V_{t+1}(R|s_{t+1}) / V_t(R|s_t) ] * log pi_R(a|s_t, R)   R ~ r(R)
   ```
   Same expected gradient as the plan's KL-to-normalised-product, at all t in one pass, with no child
   forwards. The weights are exactly ≤ 1/q = 4 when (B) holds; they are clamped at `w_max` (8).
   Cost: the (A) estimator uses `V_t(R|s)` as the normaliser instead of `sum_a pi V_{t+1}` — they differ
   by the (B) residual, which the TD loss is driving to zero anyway. Explicit children are still used
   where they are unavoidable: the posterior/EV tilt decodes in `eval.rollout_model` build 4 (×2 door
   branches) child sequences per step.
   The `a_wmean` diagnostic (mean (A) weight, should → 1) is logged so you can see when (B) is not
   yet satisfied.

2. **A smaller, open maze.** The plan's 9×9 backtracker maze with d* ∈ [20,30] and T=60 gives
   `h[0,start,-d*] ≈ (1/4)^d* ≈ 1e-12`: the random walk never produces near-optimal returns and the
   plan's own §9 rule (`h0 ≥ 1e-3`) is violated by nine orders of magnitude. Corridor mazes are the
   worst case because there is a single shortest path and every wall bump wastes a step. I scanned
   seeds of an *open room with obstacles* generator (many shortest paths) and picked a 7×7 layout with
   d*=7, T=20, `h0 = 2.0e-3`, P(reach) = 9.8 %. The backtracker generator is still in `env.py`
   (`make_maze`) for the GPU profile if you want to push d* up (expect to need T ≫ d*).

3. **Optimal episodes are removed from the data** (`drop_optimal=True`). With `h0 = 2e-3`, N=10k random
   walks contain ~20 optimal episodes, so "R = -d* never observed" would be false. Dropping them makes
   H2 a strict test: `pi_R(.|., -d*)` has zero supervised signal and must come from (A).

4. **Door maze is a small hand-parametrised layout** (`door_maze_param(a=2, b=2, depth=1, p=0.7)`;
   open route 4, around 6, locked detour 9). The plan's 10/30/12 numbers were unattainable with random
   walk data for the same `(1/4)^d` reason. See §3 for a conceptual problem with H3.

Other implementation notes:
- Grid tokens are off by default (`use_grid=False`): a fixed maze makes 121 prompt tokens pure cost.
  Turning them on is one flag and everything (index maps, rollouts) follows. Sequence length is
  `1 + 2T` = 41.
- No EOS token: nothing predicts it, so it was inert.
- `pi` in the (B) weight is the model's own NOR-mode head, as in the identity, not the known 1/4.
- Neural FQI shares the trunk and uses a scalar time-indexed Q head with a Huber loss, target network
  every 500 steps, reward −1/step and an extra −1 on timeout (so return-to-go matches R + t).
- Tabular FQI initialises unvisited entries at 0 (optimistic — the standard source of overestimation
  through coverage gaps). `fqi.biased_data` implements §9's "drop 30 % of actions".
- Every eval samples fresh random-walk prefixes, so per-t averages are automatically `d_t`-weighted
  (conditional on being alive at t), which is what the plan asks for.

## 2. Results (laptop CPU profile: d=64, 2 layers, batch 64, 2k–3k steps, 1–2 seeds; runs were
## stopped early at the user's request, so seed counts are below the plan's)

Figures: `results/E1/E1.png`, `results/E2/E2.png`, `results/E2b/E2b.png`, `results/E3/E3.png`,
`results/E4/E4.png`, `results/E5/E5.png`, `results/truth/*.png`; per-run `report.png` / `piR_arrows.png`
via `python run.py report <run_dir>`.

### E1 — H1 (MC vs TD value error): **partially supported, on the tail bins**

| N | config | TV(V_t, h_t), t<10 | |log V_t(−d*) − log h_t(−d*)|, t<4 |
|---|---|---|---|
| 2k | MC | 0.060 | 3.5 – 4.4 nats |
| 2k | TD | 0.044 | 1.6 – 2.3 nats |
| 10k | MC | 0.023 | 3.6 – 4.8 nats |
| 10k | TD | 0.029 | 2.5 nats |

- In *total variation*, TD beats MC only at N=2k; at 10k MC is slightly better for t < 10. TV is
  dominated by the 90 % "not reached" bin, which MC regression fits directly.
- On the *rare tail bin* (P(R = −d*)), TD is 1.5–2.5 nats better than MC at every N and every early t.
  This is where the H1 mechanism lives (few positives at early t for MC; TD propagates from the
  terminal), and the plan's kill criterion ("TD not below MC for t < T/2") should be read on this
  quantity, not on TV.
- Both are still 1–3 nats off on the tail bin at t = 0 (see E2b for why).

### E2 — H2 (recover pi_R at an unobserved return): **killed in the strict setting, for all configs**

| config | opt_err (nats, kill line 0.1) | P(optimal ep.) sampling pi_R(−d*) | posterior tilt |
|---|---|---|---|
| MC (plain DT / RCSL) | 0.81, 0.90 | 0.000 | 0.03 |
| TD | 0.81, 0.95 | 0.000 | 0.01 |
| TD+A (on-sequence) | 0.68, 0.82 | 0.004, 0.002 | 0.01 |

TD+A is measurably better than MC/TD on `opt_err` and on the (A) residual (0.17 vs 0.24), so the
loss is doing *something*, but it is nowhere near the kill line. Diagnosis (`logp_rstar_err`):
`V_t(R=−d* | s)` is off by 1–4 nats everywhere. With total-return bins, the bin −d* is a training target
**only** at the last token of an episode that achieved −d*. Remove those episodes and no loss touches
the bin; softmax normalisation drives it to ~0; (A) then distils a garbage posterior into pi_R.
This is not a bug in TD: TD can only propagate a terminal fact it has seen.

### E2b — what is needed to reach an unobserved bin (1 seed each, strict setting)

| variant | opt_err | logp err t=0 | P(optimal) pi_R sampling | posterior tilt sampling / greedy |
|---|---|---|---|---|
| on-seq TD+A (E2) | 0.68 | 1.7 | 0.004 | 0.016 / 0 |
| + explicit children (plan §4) | 0.69 | 0.83 | 0.006 | 0.018 / 0 |
| + rtg bins (FAIL + return-to-go) | **0.42** | **0.47** | **0.051** | **0.21 / 1.00** |
| + rtg + children | 0.46 | 0.43 | 0.045 | 0.23 / 1.00 |

- **Binning by return-to-go with a constant FAIL bin is the single most important change.** The
  terminal facts become t-independent ("at goal ⇒ 0 to go", "timed out ⇒ FAIL"), so the −d* event at
  the start is 7 backups away from a fact the model sees thousands of times, instead of a bin it never
  sees. Tail-bin error at t = 0 drops from 1.7 to 0.47 nats; the posterior-tilt greedy decode becomes
  optimal 100 % of the time and pi_R sampling is optimal 5 % of the time (from 0.4 %).
  A pure-rtg scheme (FAIL also as rtg = −(T+1−t)) trains *worse* than total bins — the dominant outcome
  then sits on a t-dependent diagonal. The plan's "own bin" remark is load-bearing.
- **Explicit children help the value tail (0.83 vs 1.7 nats) but not pi_R by themselves**, and add
  nothing on top of rtg at this budget. They are still the only way to reach (t, s) pairs the random
  walk never visited, which matters in bigger mazes.
- opt_err 0.42 is still above the kill line. The remaining gap is the transformer generalising
  "goal at t = 7" from "goal at t = 8..20" (position embeddings), plus pi_R being a softmax over a
  posterior whose mass is spread over 2–3 near-optimal actions in the open room (ties are hard to hit
  exactly in KL). The `piR_arrows.png` for `E2b/TDA_rtg_N10000_s0` is sobering: on shortest-path cells
  pi_R(−d*) is still close to uniform with only a slight down/right bias, while the DP posterior is a
  clean D/R mixture. The value tail is fixed first (0.47 nats); the policy head lags it. That is
  consistent with (A) distilling through a still-noisy `V_{t+1}(−d*|s')/V_t(−d*|s)` ratio: the
  posterior-tilt decode, which uses the value head directly, is already optimal under greedy decoding.
- The plan's *original* (rare-but-observed) setting was queued (`TDA_rare`, `MC_rare`) but not reached
  before the run was stopped.

### E3 — H3 (posterior vs EV tilt on the locked door): **model matches DP; but H3 is ill-posed as written**

DP reference (perfect values, `results/E3/dp_reference.json`):

| decode | P(route via door) |
|---|---|
| random walk | 0.59 |
| pi_R*(R=−4) / posterior tilt | **1.00** |
| EV tilt, β = 1 / 3 / 10, E under random walk | 0.50 / 0.43 / 0.79 |
| greedy on Q_opt (optimal continuation) | **0.00** |

Model (TD+A, 2 seeds): posterior tilt 0.70 / 0.68, greedy posterior 0.00, EV β=1/3/10 = 0.51 / 0.35 /
0.40, values TV = 0.026. The model's posterior tilt under-gambles (0.7 vs 1.0) for the same reason as
E2: its P(R = −4) bin is under-trained (this run used total bins, before the rtg fix).

The conceptual problem: the plan defines the EV tilt with `E[R | s, a]` **under the random walk**, so it
is just another functional of the same random-walk posterior, and on this layout it is incoherent about
the gamble (0.43–0.79 depending on β). The quantity that "doesn't gamble" is the expected return under
the *optimal continuation* (`Q_opt`: door 0.3·4 + 0.7·9 = 7.5 > around 6), which the value head does not
represent. Posterior tilt → 1.0 is robust and confirmed by the model; the "EV doesn't gamble" half of
H3 needs a controller-style value, e.g. FQI's Q head or E[R] under pi_R(R_max) rollouts.

### E4 — H4 (stability): **TD+A bounded and monotone; neural FQI slow, tabular FQI exact**

- TD+A, 2 seeds, 3000 steps: max_t value_err falls monotonically 0.98 → 0.08 (step 250) → 0.040 and stays
  flat; the per-t profile is a smooth hump (0.013 at t=0, 0.040 at t≈10, 0 at t=T), i.e. bounded and
  *sub*-linear in horizon at this scale. No divergence, seeds agree to 3 decimals.
- Neural FQI (same trunk, target net every 500 steps): max_t |max_a Q − V_opt| 7.9 → 2.7 and still
  falling in a staircase — each target update propagates the backup **one more step** from the
  terminal, so 3000 steps / 500 = 6 updates reach only 6 of the 20 steps of horizon. Overestimation is
  5.5 → 0.8 and positive throughout. Greedy decode solves 100 % but is never optimal. This is a
  budget artefact more than instability; a fair comparison needs `target_every` ≪ steps / T.
- Tabular FQI on the same 10k episodes: coverage 34 % of (t, s, a), q_err 0.097, overestimation 0.097,
  greedy optimal 100 %. This is the plan's §9 "FQI works fine" case: the maze is small and the random
  walk covers it. `--drop_actions 0.3` (implemented) was not run.

### E5 — H5: **not reached** (only the `base` run finished; closures 1–3 are implemented and smoke-tested).

## 3. Pitfalls found (beyond the plan's §9)

1. **Total-return bins cannot represent an unobserved return.** See E2/E2b. Use `[FAIL, rtg=−T..0]`.
   Corollary: "never observed in data" for a *categorical* head means "never a terminal target", which
   is a much stronger condition than "never at the start state".
2. **Pure rtg bins are worse than total bins** because the 90 % FAIL outcome becomes t-dependent.
3. **Random-walk data is hopeless in corridor mazes.** `h0 ≈ (1/4)^d*`. The maze had to be an open room
   with d* = 7 to satisfy the plan's own `h0 ≥ 1e-3`. The plan's 9×9 / d* ∈ [20, 30] / T = 60 is off by
   ~9 orders of magnitude and would need a smarter behaviour policy (and then `q ≠` random walk, which
   is exactly closure 1).
4. **The EV tilt is not a controller** unless its expectation is under a controller (E3).
5. **TV is the wrong headline metric.** It is dominated by the FAIL bin; the tail-bin log error
   (`logp_rstar_err`) is where MC and TD differ and where H2 is decided.
6. **Posterior-tilt decode needs a fallback.** Where V(R* | s, a) ≈ 0 for all a the tilt is 0/0;
   `rollout_model` falls back to pi. With a wrong tail bin this fallback is silently what runs.
7. **Neural FQI with a target network is throughput-bound by `steps / target_every ≥ T`.**
8. **Greedy pi_R decode (argmax) reached 100 % solve rate with 0 % optimal** in several runs — it
   follows the mode of a posterior that was trained on failures. Sampling is the honest decode.
9. `nan` hygiene: masked means must use `where(mask, x, 0)`, not `x * mask`; KL with `p = 0` must be
   masked before the log.

## 4. Suggested tweaks (in the order I would try them)

1. **Default to `rtg=True`** (done as a flag; switch the default) and re-run E1–E3 with it; E3 in
   particular should reach posterior door_choice ≈ 1.0.
2. **Analytic terminal on children + a `children`-only TD term for the last step** — cheap and makes
   the base case exact by construction.
3. **Larger model / longer runs for the generalisation gap in E2b**: the residual 0.42 nats is a
   capacity/position-embedding question. Try relative or sinusoidal time embeddings so "goal at t"
   generalises across t trivially; or feed `t` as an explicit token.
4. **For H3, add a controller-EV decode**: `pi(a) exp(β Q_FQI(s,a))` from the neural FQI head, or
   `E[R]` under pi_R(R_max) rollouts. The DP reference for it is `greedy_opt` (0.00).
5. **Behaviour policy other than the random walk** to use the plan's bigger maze: e.g. ε-greedy on a
   noisy DP policy, and log `logq` (already a field) so the importance weights stay exact.
6. **E4 fairness**: `target_every = 50`, 20k steps, and `--drop_actions 0.3` to create the coverage
   gaps §9 asks for.
7. **Weight r(R) by the model's own uncertainty** rather than a fixed 3× on the top bins: the (A)
   signal is zero wherever V(R | s) ≈ 0, and `a_child_frac` (≈0.5) shows half the child (A) targets are
   undefined.
8. **Report `opt_err` as a JS-divergence or top-action agreement** in addition to KL; KL against a
   uniform-over-ties target punishes small mass imbalances harshly.

## 5. Quality-of-life code

`maze_consistency/viz.py`: `plot_maze` (walls / start / goal / door, per-cell heatmap with log scale,
policy arrows scaled by probability, path overlay), `plot_visitation`, `plot_per_t`, `plot_vs_step`,
`plot_losses`, `plot_ground_truth` (four-panel DP overview), `run_report` (12-panel per-run figure +
DP-vs-model pi_R arrows), `print_rollout_table`. `python run.py report <run_dir>` and
`python run.py plot E<k>` regenerate everything from `metrics.json`; `python run.py truth` renders the
DP truth for both mazes.
