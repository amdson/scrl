# Model-based rollout augmentation: status

**State:** the current revision removes oracle information from training. Previous versions (`b990a16`,
`be0f34e`) used real transitions or exact shortest-path distances to generate/select training data. Those
paths are now removed or refused. Measurements below describe the old selection rule; rerun with the new
`rollouts_offline` notebook prefix before comparing results.

**Goal:** generate optimal paths by learning from non-optimal offline trajectories. Training must discover
useful compositions itself; no optimal-path selection, exact-distance feasibility checks, or real-maze
interaction supplies its targets. The real maze is available for evaluation only.

## 1. Why

Two results from the consistency and enrichment experiments motivate this.

**Conditioning on reward works, weakly.** At the identical prefix, asking for the best achievable bin
instead of bin 0 raises the probability of a goalward move. The exact conditional gains ~39.5 points. At 10k
steps the trained models gain:

| run | gain (points) | share of exact |
|---|---|---|
| `mc` | ~1.8 | ~4.6% |
| `mc_local` | ~3.9 | ~9.9% |
| `mc_all` | ~4.9 | ~12.4% |

Consistency gives a stable ~2.5x multiplier over `mc` from 3k to 10k steps. Almost the entire deficit is on
the "ask best" side: the models ask-fail at ~32.5% goalward (exact: 31.5%) and ask-best at ~37% (exact:
71.3%). They can't produce goal-seeking behaviour when asked for it.

**The data doesn't contain that behaviour.** Of 66,609 far-start random walks (start >= 10 steps from the
goal), 5,420 reach the goal and **0** achieve their start's best outcome bin. Nothing in the data demonstrates
near-optimal play from far starts, so there is nothing to imitate.

The augmentation is meant to manufacture that missing behaviour offline.

## 2. What it does

For a training trajectory:

1. Pick a switch time `tau` and keep the real prefix `h_tau`.
2. Continue from `tau` *in imagination*. Actions come from the model's reward-conditioned policy, asking for
   a bin above the one the original trajectory achieved. Next cells come from the model's own dynamics head.
3. Relabel the spliced trajectory with the bin it actually achieved.
4. Feed it back into training: teacher forcing in both modes, the MC value head, and the consistency term.

A `RolloutBuffer` holds the spliced trajectories and regenerates them from the current model every
`REFRESH_EVERY` steps once training passes `START_AFTER`.

## 3. Design decisions

**Continuations are imagined.** `splice` requires learned `cell_logits`; `RolloutBuffer(dynamics="env")`
is refused. Real-maze rollouts remain in `evaluate.py` for evaluation and must not enter training.

**Requests do not use optimal distances.** `request="above"` samples uniformly from bins above the recorded
outcome through K-1. `request="highest"` always asks for K-1 (`"best"` is a legacy alias with this new meaning).
Only prefixes already finished or rows already labelled K-1 are resampled. Requests may be infeasible; all
generated outcomes are kept. No distance or reachability calculation clips requests or selects training rows.

**Next cells come from the dynamics head read in NOR mode**, the unconditioned world model `p(s' | h, a)`.
Conditioning the dynamics on a high requested reward would bias it toward lucky transitions
(`consistency_losses.md` section 7), and asking for high reward selects for exactly those.

**The real maze is used only for evaluation.** After generation, `evaluate.rollout_diagnostics` counts
impossible moves, computes exact random-walk baselines, and measures far-start best-bin attainment. These
results never filter or modify training rows. Generation uses task definitions (dimensions, goal, horizon,
reward bins), without consulting the maze layout, transition table, or optimal-distance map.

**Relabel to the achieved bin, never the requested one.** Asking shapes which states get visited; the label
follows what happened.

**Every head trains on the mixture.** The interval identity `p(x | h, R) q_i(R) = p(x | h) q_j(R)` is a
statement about one joint distribution. If only the reward-conditioned head saw rollouts while the NOR and
value heads kept the random walk, the heads would model different joints and the consistency loss would fight
the data. `train(mixer=...)` shuffles rollout rows into both halves of every batch and into the consistency
batch.

**`tau` is independent of the trajectory's future**: `tau ~ U{0..TAU_MAX}`, keeping rows still running at
`tau`. Drawing `tau ~ U(0, L)` would make the switch point depend on how long the original ran.

**`td` is refused with a mixer.** Its importance weight assumes the uniform random-walk behaviour policy,
which rollout rows don't follow. Use `mc` with optional interval consistency. The separate legacy `a`
objective built child states from real transitions and is disabled, even without a mixer.
`train(consistency=True)` now selects MC plus interval consistency.

## 4. Code

| where | what |
|---|---|
| `evaluate.continue_rollout` | continues trajectories from per-row prefixes; imagined dynamics when given `cell_logits` |
| `evaluate.make_cell_logits` | next-cell logits at an action slot (the world model) |
| `evaluate._trajectory` | reads a token buffer back into dataset layout; shared with `rollout` |
| `augment.splice` | picks recorded prefixes, requests a higher configured bin, imagines and relabels |
| `evaluate.rollout_diagnostics` | evaluates completed splices with exact transitions, distances, and DP baselines |
| `augment.summarize` | per-refresh behavioural readout (section 6) |
| `augment.RolloutBuffer` | the pool; duck-typed for `train(mixer=...)` |
| `train(mixer=, mix_frac=)`, `merge_rows` | mixes rollout rows into the main and consistency batches |
| `tests/test_augment.py` | generation, evaluation, offline-boundary, and training tests (section 5) |
| `colab/rollouts.ipynb` | the 2x2 experiment (section 6) |

With no mixer, training is unchanged: a baseline run reproduces `mc 1.3715 tf 4.8714` exactly.

## 5. Validation

- **No oracle access in training.** A guarded maze raises on exact distances, transitions, layout, optimal
  bins, or start-cell enumeration. Imagined generation and mixed MC/consistency training pass with that
  guard; only separate diagnostics receive the real maze.
- **Infeasible requests and failures are kept.** Requests can exceed the exact feasible bin. A fake world
  model that never reaches the goal still supplies every requested training row. Diagnostics leave those
  rows unchanged. Environment-backed buffers and the legacy oracle-child objective are refused.
- The numerical smoke results below are historical, before oracle selection was removed.

- **Prefix kept, dynamics correct.** The first `tau` steps are copied verbatim, and env-stepped continuations
  follow `maze.next_open`.
- **The world model is read at the right slot.** An oracle world model (one-hot on the true next cell)
  reproduces the real maze with zero impossible moves. For the real model, `n_bad` matches a hand count of
  impossible moves after `tau`.
- **End-to-end against the exact answer.** With a uniform policy in place of the model, a continuation is a
  random walk, so its rates must match the DP's exact per-row baselines. It does, under both the real maze
  and the oracle world model (800 rows):

  | | measured | exact random walk | 4 s.e. |
  |---|---|---|---|
  | beat the original continuation | 0.114 | 0.107 | 0.040 |
  | reached the requested bin | 0.061 | 0.059 | 0.030 |

- **Relabelling, the refresh schedule, and the `td` refusal** are each tested.
- **The hallucination diagnostic catches what it should.** In a smoke run with an untrained 4-step model and
  an 8-row buffer, rollouts "improved" 75% of the time (random walk: 8%), and 100% of them contained an
  impossible move.

## 6. The experiment: `colab/rollouts.ipynb`

A 2x2, `{mc, mc_all} x {data only, + rollouts}`, so the rollout effect and its interaction with consistency
both show.

| parameter | default | note |
|---|---|---|
| `STEPS` / `BATCH` | 20000 / 32 | try 3000 first; GPU cost is unmeasured |
| `D_MODEL` / `N_LAYERS` | 64 / 2 | same as the 10k runs, so `mc_all`'s lambda (0.011) still applies |
| `MIX_FRAC` | 0.25 | share of each main and consistency batch from the buffer |
| `BUFFER_SIZE` / `REFRESH_EVERY` | 512 / 500 | ~400 forward passes per refresh (one per action, one per next cell) |
| `START_AFTER` | 2000 | |
| `TAU_MAX` | 100 | see section 9 |
| `REQUEST` | `"above"` | or `"highest"`; neither uses feasibility information |
| dynamics | learned NOR model | required for training; real maze only for evaluation |

**Read section 3 of the notebook first.** Per refresh, from real prefixes, it shows how often the model:

- beats the original continuation's bin,
- reaches at least the requested bin,
- reaches the goal at all,
- produces far-start trajectories hitting their own best bin (the data has 0),
- hallucinates, overall and among the rollouts that *improved*.

Improvement and request-attainment rates are compared with exact random-walk rates from the same prefixes.
These are imagined outcomes, not proof of improved real-maze behavior. High-reward requests can exploit
world-model errors, so
if hallucinations among improved rollouts run well above the overall rate, the loop is training on
hallucinated shortcuts.

**Other metrics, read with care.** The test set's `act_kl` / `value_kl` compare against the random walk.
Rollout-trained runs deliberately learn a different distribution, so worse numbers there are expected and not
a failure. Enrichment *points* stay meaningful; the "% of exact" column can exceed 100%.

**Expected world-model quality.** Per-step dynamics accuracy was 99.49% (cons) to 99.97% (base) at 2000
steps, i.e. roughly 86-99% of 30-step imagined rollouts free of impossible moves, before the selection effect
above.

## 7. Data accounting

Historical accounting before oracle selection was removed, from step 2000 on:

| | real | synthetic | ratio |
|---|---|---|---|
| rows per main batch | 24 | 8 | 3 : 1 |
| rows per consistency batch | 12 | 4 | 3 : 1 |
| rows trained on, whole run | ~496k | ~144k | 3.4 : 1 |
| unique trajectories | 99k | ~18.9k (37 refreshes x 512) | 5.2 : 1 |
| reuse | ~5x each over 20k steps | ~8x main + ~4x consistency within one 500-step window | |

Only part of a spliced row is synthetic: the real prefix averages 49 steps.

## 8. Known issue: successful rollouts get the least loss weight

The next-token and MC losses average over all tokens in the batch, not per trajectory, so long rows count for
more. Real rows average 172 steps (78% time out at 200). The synthetic share of those losses is then:

- **~2%** if continuations reach the goal near-optimally (~61 steps total, ~12 imagined), and
- **~21%** if continuations time out at 200 steps.

The successful rollouts, the ones carrying the behaviour the data lacks, get the least weight, and they get
less as the model improves. The consistency term averages per trajectory first, so there synthetic rows get
their full 25%.

**Fix (not implemented):** average the next-token and MC losses per trajectory, or weight spliced rows up.
Per-trajectory averaging also re-weights real rows against each other, so it changes the baseline slightly.

## 9. Known issue: few splices can produce the missing behaviour

Historical measurements for the removed oracle-feasibility selection rule, without a model. Remeasure these
for the current rule; exact distances may evaluate the generated pool, but must not select it.

| | kept splices | all training data |
|---|---|---|
| draws kept by the filter | 89% | — |
| original trajectory failed | 87% | 78% |
| far start | 74% | 67% |
| `tau = 0` (bare start) | 1% | — |
| `tau < 10` | 11% | — |
| far start that can still hit its own best bin | **4.9%** | — |

The target behaviour, far starts reaching their own best bin, is possible in only 4.9% of splices, about 25
per 512-row refresh. From a far start, a random-walk prefix of `tau` steps has usually already wasted too
many steps.

**Fix (not implemented):** lower `TAU_MAX` (e.g. 10) or weight `tau` toward 0. The cost is fewer
mid-trajectory prefixes, so less variety in the states the model continues from.

## 10. Next steps

1. Decide on sections 8 and 9. Both are small changes, and both push the run toward the behaviour it's
   meant to produce.
2. Run the notebook at `STEPS=3000` to measure GPU time per step and per refresh.
3. Run the full 2x2. Check the notebook's section 3 first: is "improved" pulling away from the random-walk
   baseline, is the far-start own-best count nonzero, and do hallucinations among improved rollouts stay close
   to the overall rate?
4. Evaluate the learned policy in the real maze from fixed starts, including optimal-path attainment.
   Compare real and imagined performance to measure world-model error. Never train on evaluation paths.

Related, not part of this change: with the default uniform binning, almost all the conditioning signal sits
in bins 10-11. Geometric binning (`Maze(binning="geometric")`, `colab/binning.ipynb`) spreads it over ~7 bins.
The rollouts notebook still uses uniform bins.
