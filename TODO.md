# TODO: consistency with model-proposed queries (writeup/math.tex)

Rule: training uses the recorded dataset only. No maze walls, goal cell, reward evaluation or bin
feasibility in training or generation. The real maze is for evaluation only. No relabel fitting.

## 1. Tilted query sampler on data rows -- DONE (`consistency.make_tilted_sampler`)

A `cons_sampler(params, rng, n, step)` for `train()`:
- take a batch of recorded rows, run one NOR forward pass, read the reward head at the start slot (q_0)
- tilt: rho(k) ~ q_0(k) * exp(beta(step) * r_k), r_k = maze.bin_reward (label definitions)
- sample one query bin per row; build the conditioned tokens with that bin instead of the recorded one
- beta follows a schedule on the step
- the residual is then enforced at (prefix, query bin) pairs the data never contains; no generation needed
- tested 2026-09-15 from the `cons` checkpoint, 500 steps, beta ramp to 10: enrichment 3.1% -> 3.6% of the
  ceiling (noise level), and the far-start value tail got WORSE (q0(bin 11) at dist 18: 5.9e-3 -> 1.4e-2,
  exact 1.8e-11). Mechanism: with query bin 11 on full random-walk rows, 99.8% of interval pairs lie past
  feasibility; the residual is satisfied by raising the start belief instead of lowering the end belief.
- windowing intervals by the bin's arrival time is NOT allowed (bin edges are environment knowledge)
- allowed fix: `truncate=True` cuts intervals where the model's own belief in the query bin < `floor`

## 2. Learned-termination generation, rollouts as consistency rows only -- DONE (`make_rollout_sampler`)

- `continue_rollout`: sample from actions + END at state slots; END ends the row; never consult the goal
- a second sampler: splice real prefixes, request a bin from the tilted head at the switch point (or the
  top bin), continue in imagination with NOR dynamics, return the rows as a consistency batch with the
  request as the query
- no relabelling; rows never enter the main (likelihood) batch
- `continue_rollout(learned_end=True)` samples actions + END; `make_rollout_sampler` builds the batch from the
  token buffer with query = request; `make_phased_sampler` switches from recorded bins at a start step
- still to do: move the goal / achieved-bin computations in `augment.splice` into diagnostics (the old
  relabelling path is unused but still present)

## 3. Evaluation-side diagnostics (may use the maze) -- partly DONE

- DONE: `evaluate.imagined_rollout_eval` (invalid imagined transitions, END at goal, reached) -- eval only,
  called from the notebook logger; `train(grad_every=)` logs per-term gradient norms with the consistency
  term split recorded vs proposal rows (`gnorm/*`, `cons_rows/*`); W&B logging via `train(metrics_fn=)`
- early reading (2k-step checkpoint, 10-step rollouts): proposal rows' consistency gradient norm 10-400x the
  recorded rows'; per-row loss 3-5 +- 4-9 vs 0.03-0.3. If this holds in the long run, weight proposal rows
  down (per-row weight or a lower lambda for them) rather than raising KEEP_RECORDED
- fraction of query bins the DP says are infeasible at the prefix
- residual size by query bin
- hallucination rate on proposal rows
- existing: cond_gap / info_gain collapse checks, enrichment vs the exact ceiling, goalward when asking
  for the best bin, real-maze return from far starts

## Next run: `colab/late_rollouts.ipynb`

Offline phase first, then model-proposed queries switched on at `START_LATE` (hypothesis: a longer offline
phase gives the value head a real far-start tail, so the belief floor can work). Arms `mc_all`, `late_tilt`,
`late_rollout`, `late_both`; small constant beta; readouts: enrichment, far-start q0 tail by distance,
sampler histograms and floor cuts. W&B logging (`metrics_fn`), per-term gradient norms (`grad_every`),
real-maze rollout eval, and mid-run checkpoints (`ckpt_every`, resume on restart) are wired in.

## Experiments

First Colab run (uniform K=12, 20k steps, late switch at 12k): no benefit from either proposal source; all
arms improved steadily on value KL / enrichment through the whole run with tf at its entropy floor, i.e. the
bottleneck is propagation of value into rare regions, not token modelling.

Now switched to **geometric binning, K=24** (`MAZE_KW` in the notebook; own exact test set
`data/canonical/testset_geo24.npz`): optimal play spans 11 usable bins (far starts: bins 10-14) instead of 2.
Bins 18, 21, 22 are empty. Panels/tables read the far-start best bins (`FAR_BINS`), not the top bins.

Speed sweep in the notebook (`SPEED_SPECS`, 5k steps, `RUN_SPEED_SWEEP`): constant 1e-3 vs warmup+cosine 3e-3,
model 128x4, consistency batch 32, lambda 0.5. `train()` now takes warmup / cosine / lr_end_frac; a spec may
carry per-run train kwargs. Still untried: all-interval/multiscale vs local at scale, the one-endpoint
stop-gradient ablation (faster one-way propagation; deadly-triad setting). Watch held-out value KL vs train
MC for value-head memorization on long runs (>6 epochs over 100k rows).

Rollouts: gate the late switch on goalward/best rather than step count; normalize the proposal term by its own
gradient norm; select rollout rows by the model's terminal belief in the request (a legal proposal choice).
Alternative conditioning if per-bin sparsity persists: threshold events "arrived within H" (cumulative bins,
same head, no architecture change).

## Doc

- goal statement up front: target is the data process's own conditional; the DP is the evaluation ceiling
- experiment section: arms, metrics, and what counts as the identity doing work the baseline cannot
