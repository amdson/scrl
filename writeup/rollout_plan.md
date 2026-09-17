# Rollout training: the proposed framework

A plain statement of what I am proposing we test, written so each piece can be accepted or rejected on its own.
Notation follows `math.tex`: a prefix is `h_i`, a continuation is `x`, the query reward event is `r`, and the
model's NOR belief in `r` at prefix `h_i` is `q_i(r)`. Under threshold conditioning `r` is the event "bin k or
faster", and `q_i(r)` is a tail sum of the value head.

## 1. What is trained today (the base)

The current best arm (`mc` plus consistency with the `all` objective, threshold, RoPE, ordinal modes) minimises, on recorded random-walk rows only:

- **Teacher forcing.** Next-token likelihood of recorded actions, states and END, in NOR mode and in the
  conditioned mode with the row's own satisfied thresholds.
- **Value head.** Likelihood of the recorded outcome bin at every prefix (the `mc` term).
- **Consistency on recorded rows.** The interval residual

      Delta_{i,j}(r) = [log p(x | h_i, r) - log p(x | h_i)] + [log q_i(r) - log q_j(r)]

  over all intervals `i < j` of a recorded row (the `all` objective, the unscaled variance shortcut), with `r` drawn from the
  pairs (row, satisfied threshold). Both token-probability terms include the action slot and the state slot.

Everything below adds rows to the consistency term. Nothing else changes: teacher forcing and the value-head
term stay on recorded rows only, and no imagined token or imagined reward is ever fitted as data.

## 2. Why recorded rows are not enough

Recorded rows contain every interval a random walk produces. What they almost never contain is a far prefix
followed by ten goal-directed steps. That interval is exactly what the test asks the conditioned policy to
produce. The identity constrains `p(x | h_i, r)` for goal-directed `x` only where such an `x` is presented, so a
proposal must present it. Under threshold conditioning with geometric bins, the random-walk odds of a start's own
best event are about 1e-2 at distance 3, 2e-6 at distance 10 and 1e-9 at distance 20, and two bins slower is
about a hundred times likelier at every distance. That is the ladder the proposals have to climb.

## 3. One proposal row, step by step

A refresh produces a buffer of rows. Each row is built as follows.

1. **Prefix.** A recorded row cut at a uniform `tau` in `[0, length)`, so the prefix is a real random-walk
   history ending at some cell, possibly far from the goal, possibly after many steps. Both matter: the event
   is about the whole episode's arrival time, so a prefix that has already spent fifty steps at a cell has a
   different tail belief than a fresh start there, and no recorded row ever follows such a history with
   goal-directed steps. Late prefixes are low-information (a prefix that has spent most of the horizon can
   only satisfy the slowest thresholds), but uniform `tau` is the simplest choice and the rows still show
   whether the mechanism works; shaping `tau` is ablation 3. (The current sampler uses `tau = 0` only because
   its belief-floor cut, removed below, had to land after the imagined steps.)
2. **Request.** Read the NOR value head at the prefix `h_tau` and form the tail beliefs `q_tau(k)` for every
   threshold `k >= 1`. The request `c` is the most ambitious threshold the model still gives probability at
   least `p`: the largest `k` with `q_tau(k) >= p`. Since tail beliefs are non-increasing in `k` this is a
   single quantile of the value head, adapts to the prefix (a near-goal prefix asks for a fast arrival, a far
   prefix for whatever it can still believe in), and uses no environment knowledge. `p` is the one knob, and
   lowering it over training is the schedule (section 5). No global request such as "the highest bin".
   The exponential tilt `q_tau(k) * exp(beta r_k)` stays as an arm, not the default: under geometric binning
   the bin rewards relevant to far prefixes are 0.03 to 0.3, so one `beta` cannot lift a far event without
   saturating every near prefix at the top bin (lifting a 1e-9 event over that gap needs `beta` near 70).
3. **Continuation.** Ten steps at most. Actions from the conditioned policy under `c`, next cells from the
   dynamics head read in NOR mode (as `continue_rollout` already does), and the model may emit END. No maze
   walls, goal or distances are consulted.
4. **No filtering.** Every row is kept whole, whatever the model believes about `c` at the prefix or at the
   endpoint. The current sampler cuts a row at the first prefix where the NOR belief in `c` falls below
   `FLOOR = 1e-3` (`make_belief_truncation`); that rule is removed. The residual lives in log space and its
   gradient does not shrink as beliefs shrink, so a small belief is not a problem in itself, and at a far
   prefix the belief in an ambitious request is supposed to be small: that is the row that stitches. The one
   thing a filter could buy is dropping rows where neither end carries information; whether such rows matter
   is an empirical question, answered by the belief-ratio readout in section 6.
5. **Query.** The residual is evaluated with `r = c` on the intervals inside the imagined segment only: all
   `tau <= i < j <= tau + 10`, with the recorded prefix serving as context, not as intervals. This needs a
   per-row start in `consistency.residuals` (blocks before `tau` masked, so the cumulative residual starts at
   `tau`). Enforcing the whole row instead would be wrong for the `all` objective: the variance shortcut weights
   step `t` by about `t (n - t)`, so with a long prefix the ten imagined steps at the end of the row carry the
   smallest weight in it and the loss is dominated by intervals lying entirely inside the recorded prefix,
   which are just recorded-row consistency under query `c`.

What a row does: its endpoint is nearer the goal (if the conditioned policy is any good), where the belief in
`c` is larger and better trained; its prefix belief may be tiny. The residual ties the prefix belief and the
conditioned policy's probability of the continuation to that endpoint belief. That is the stitching step.

## 4. Loss and weights

- The consistency batch is a fixed mixture: a `keep_recorded` share of recorded rows with threshold-pair
  queries (the base), and the rest proposal rows. Start at half and half.
- Proposal rows get their own weight `lambda_prop`, separate from the recorded rows' `lambda`. On the old
  sampler's rows (one-step rows under an impossible request) the gradient norms ran 10 to 400 times those of
  recorded rows; that is not a measurement of these rows, only a reason to start `lambda_prop` small, around a
  tenth of `lambda`, and raise it if the far-start metrics move without the teacher-forcing loss rising.
- No clipping for now. The per-term gradient norms already logged every 100 steps (`gnorm/cons_proposal`
  against `gnorm/cons_recorded` and `gnorm/tf`) are the monitor; if proposal rows dominate a step, the first
  response is a smaller `lambda_prop`, and clipping of the proposal term's gradient is the fallback.
- Full gradients, no stop-gradients, as now.

## 5. Schedules

- **Introduce proposals late.** After the base has trained the value head near the goal, since that is the
  anchor the rows propagate from. With the two-stage process of section 5a this is automatic: proposals are
  on from the first step of stage 2.
- **The request quantile `p`.** Start around 0.1, so the request is an event the model believes in but is
  not the likely outcome, and lower it over training (for example by a fixed factor every refresh, or by
  hand between arms). A too-low `p` produces rows whose request the policy cannot make progress on; the
  belief-ratio readout shows this as ratios near one, and is the signal to slow the schedule. At `p` near
  one the request is "reached at all" and the rows are random walks the recorded rows already cover, so
  there is no point starting there.
- **Refresh.** Regenerate the buffer every few hundred steps so the frontier moves with the model.

## 5a. Two-stage training: the notebook's shape

Many rollout configurations will be tried, and each should start from the same model that is already known
to be a decent world model. So the notebook is built as two stages, not one run with a phase switch.

**Stage 1, the base checkpoint.** One run of teacher forcing, `mc`, and consistency on recorded rows (the
base of section 1) for about 5000 steps, saved under a name that records everything it depends on: binning,
conditioning, positional and mode encodings, model size. It is trained once and reused. Its final evaluation
is the common starting point every stage-2 arm is measured from.

**Stage 2, the rollout arms.** Each configuration is a separate run started from the stage-1 parameters
(`train(init_params=...)`, which already exists; the optimizer state is fresh, so a short warmup is needed)
with proposals on from its first step. The `START_LATE` phase switch is then unnecessary: "late" is simply
"stage 2". Every arm runs the same number of steps and is evaluated at step 0, so the shared baseline is in
every arm's history.

**The control arm.** One stage-2 arm continues the base objective with no proposals for the same number of
steps, restarted from the same checkpoint with the same fresh optimizer and warmup as every other arm.
Improvements are measured against it, not against the frozen checkpoint, because the base keeps improving on
its own and the question is whether proposals add anything at equal compute.

Practical consequences: the stage-1 name is a notebook parameter and stage 1 is skipped when it exists; each
stage-2 arm is a small dict of sampler and loss settings, so a sweep is a list of dicts; the world-model
probes (start-value error, invalid-step fraction) are run on the base checkpoint once, as the reference the
arms must not degrade.

## 6. What to log, and what counts as working

Per refresh:

- The distribution of endpoint beliefs `q_j(c)` and prefix beliefs `q_tau(c)`, both in log space. The ratio
  `q_j(c) / q_tau(c)` is the propagation signal: a row with a ratio near one is redundant with recorded rows,
  and if such rows dominate the batch, that is the case for adding an endpoint filter.
- Prefix-distance histogram. If rows with a large belief ratio cluster near the goal, add explicit frontier
  selection by prefix belief (section 8).
- Requested-threshold histogram, and the request quantile `p` in force.
- (The belief readouts need one extra NOR forward pass over the buffer per refresh to read the value head at
  the prefix and the endpoint; nothing else computes it.)
- **Invalid transitions in the training rollouts.** Every refresh's buffer is checked against the real maze
  (evaluation only, nothing feeds back): the fraction of imagined steps whose next cell is not what the walls
  allow, split by kind (wrong direction, wall, teleport), the fraction of rows with at least one such step,
  and the END-at-goal rate. This is the existing `rollout_eval/*` readout from `imagined_rollout_eval` on the
  sampler's last buffer, extended with the kind split, logged every refresh so it forms a curve over training.
  No separate probe rollouts are generated. The control arm generates no rollouts, so it has no curve; the
  base checkpoint's value from the world-model probe notebook is the reference instead. An arm whose curve
  rises above that reference is buying its reward gains with a broken world model, and that is a failed
  configuration whatever its other metrics say.

Per eval:

- Far-start value KL and the start-value error curve from the world-model probe.
- Held-out teacher-forcing loss on recorded rows. Rising means the proposal rows are leaking into on-support
  predictions.
- Conditioned rollouts in the real maze from far starts, scored on the achieved event (the probe notebook).

Working means: the far-start conditioned rollouts achieve their requested event more often than the control
arm at equal steps, while the invalid-step fraction and the held-out teacher-forcing loss stay flat.

## 7. What the previous rollout arm actually tested

The run that showed no benefit requested bin 23 ("arrive within 1.26 steps") from every start under geometric
binning; two of seventy-two starts can satisfy it. The belief-floor cut then landed at the start prefix, which
cut nearly every row to a single step. So the arm enforced the identity on one-step rows conditioned on an event
the model rightly disbelieved. It was not a test of stitching, and it says nothing about section 3.

## 8. Ablations and open questions, in order of priority

1. **Stop-gradient on the endpoint belief.** Full gradients spread the residual over all four terms and may
   propagate slowly from the endpoint inward. Detaching `log q_j(c)` makes the update directional from the
   anchor (semi-gradient TD in the log domain). It is the first thing to try if far-start beliefs stay flat.
   It is well defined for the local one-step loss, not for the variance shortcut, so this ablation also
   switches proposal rows to the local loss.
2. **Explicit frontier prefixes.** Prefer prefixes whose belief in the request lies in a middle band, so the
   buffer is not dominated by near-goal prefixes. Only if the prefix-distance readout shows that.
3. **Prefix placement.** Cap `tau` at half the horizon, or weight early prefixes, so fewer rows are spent on
   prefixes that can only satisfy the slowest thresholds. Only if the prefix readouts show most rows are such.
4. **Endpoint filter.** Drop rows whose endpoint belief is below some small threshold, if rows with a belief
   ratio near one turn out to dominate. The threshold would mark whether the endpoint is a learned anchor,
   not an absolute probability; the old 1e-3 is far too high for that.
5. **Residual without the dynamics factor.** Read the state-slot probability in NOR mode on both sides of the
   residual so it cancels. This closes a channel where a value-head inconsistency can be absorbed into a
   conditioned-versus-NOR dynamics ratio. Deferred: the simpler residual first.
6. **Counterfactual queries on recorded rows.** A third mixture component: recorded rows with the query drawn
   by the same quantile rule at their start (the existing tilted sampler, without truncation). It costs no
   rollouts and gives off-policy coverage of wandering continuations under ambitious queries, which is what
   forces the conditioned policy to disfavour wandering rather than merely lowering its probability of the
   goal-directed path. The self-sampled rollouts probably cover this through normalisation; this is the hedge.
7. **Early start.** Proposals from step zero of a fresh model, to see whether the base term alone regulates them.

## 9. Where it lives (implemented 2026-09-17)

- `consistency.residuals(u, v, b, lengths, starts)`: per-row start; blocks before `starts` are context only,
  every loss and the diagnostics read the segment `[starts, lengths]`. `rollout_batch` and the tilted
  sampler emit `starts = 0`; the rollout sampler emits `starts = tau`.
- `consistency.make_rollout_sampler(request="quantile", p=...)`: uniform `tau`, quantile request
  (`_quantile_query`), no floor, at most `max_steps` imagined steps, readouts in `sampler.history` and
  `sampler.last_readout()` (prefix and endpoint log-belief in the request); `request="tilt"` and `"highest"`
  remain as arms.
- `train.LossConfig.w_prop`: the proposal rows' weight (rows after the sampler's `n_rec`); `cons_rec`,
  `cons_prop` and the weighted term `cons_w` are logged. No clipping beyond the existing global norm.
- `evaluate.imagined_rollout_eval`: adds `wrong_dir_frac`, `wall_frac`, `teleport_frac`.
- Defaults: `train(cond="threshold", pos_enc="rope", mode_enc="ordinal")` and the same in `ModelConfig`;
  `load_run` fills the original values for runs saved before those options existed, and `run_cond` reads a
  run's conditioning for `run.py eval`.
- `colab/rollout_training.ipynb`: the two-stage notebook (base run, arms as dicts, control arm, W&B logger
  with the readouts of section 6, curves). Run locally it executes a tiny smoke configuration.
