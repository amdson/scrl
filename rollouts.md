# Model-based rollout augmentation: status

**State:** implemented, tested, pushed to `main` (`b990a16`, then `be0f34e`). Not yet run at scale. Two
known issues (sections 8 and 9) are worth deciding before the scaled run, because both shrink the effect the
run is meant to measure.

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

**Continuations are imagined, not stepped in the real maze.** The first version (`b990a16`) stepped the real
maze for the continuations. That's online interaction, i.e. on-policy data collection. It isn't model-based,
and it isn't the offline setting the rest of the project works in. `be0f34e` fixed this. `dynamics="env"`
still exists, explicitly labelled as an online upper bound for measuring how much the world model's errors
cost.

**Next cells come from the dynamics head read in NOR mode**, the unconditioned world model `p(s' | h, a)`.
Conditioning the dynamics on a high requested reward would bias it toward lucky transitions
(`consistency_losses.md` section 7), and asking for high reward selects for exactly those.

**The real maze is used only as a diagnostic.** It counts imagined moves it wouldn't allow (`n_bad`) and never
produces data.

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
which rollout rows don't follow. `mc` and identity (A) are fine.

## 4. Code

| where | what |
|---|---|
| `evaluate.continue_rollout` | continues trajectories from per-row prefixes; imagined dynamics when given `cell_logits` |
| `evaluate.make_cell_logits` | next-cell logits at an action slot (the world model) |
| `evaluate._trajectory` | reads a token buffer back into dataset layout; shared with `rollout` |
| `augment.splice` | picks prefixes, requests a higher bin, continues, relabels, records exact random-walk baselines |
| `augment.summarize` | per-refresh behavioural readout (section 6) |
| `augment.RolloutBuffer` | the pool; duck-typed for `train(mixer=...)` |
| `train(mixer=, mix_frac=)`, `merge_rows` | mixes rollout rows into the main and consistency batches |
| `tests/test_augment.py` | 5 tests (section 5) |
| `colab/rollouts.ipynb` | the 2x2 experiment (section 6) |

With no mixer, training is unchanged: a baseline run reproduces `mc 1.3715 tf 4.8714` exactly.

## 5. Validation

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
| `REQUEST` | `"above"` | or `"best"` |
| `DYNAMICS` | `"model"` | `"env"` = online upper bound |

**Read section 3 of the notebook first.** Per refresh, from real prefixes, it shows how often the model:

- beats the original continuation's bin,
- reaches at least the requested bin,
- reaches the goal at all,
- produces far-start trajectories hitting their own best bin (the data has 0),
- hallucinates, overall and among the rollouts that *improved*.

Each rate is plotted against the exact random-walk rate from the same prefixes (from the DP), so it stays
valid however training shifts the model. Asking for high reward selects for world-model errors that help, so
if hallucinations among improved rollouts run well above the overall rate, the loop is training on
hallucinated shortcuts.

**Other metrics, read with care.** The test set's `act_kl` / `value_kl` compare against the random walk.
Rollout-trained runs deliberately learn a different distribution, so worse numbers there are expected and not
a failure. Enrichment *points* stay meaningful; the "% of exact" column can exceed 100%.

**Expected world-model quality.** Per-step dynamics accuracy was 99.49% (cons) to 99.97% (base) at 2000
steps, i.e. roughly 86-99% of 30-step imagined rollouts free of impossible moves, before the selection effect
above.

## 7. Data accounting

With the defaults, from step 2000 on:

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

What the selection rule actually picks, measured without a model:

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
4. If it bootstraps, compare against `DYNAMICS="env"` to see what the world model's errors cost.

Related, not part of this change: with the default uniform binning, almost all the conditioning signal sits
in bins 10-11. Geometric binning (`Maze(binning="geometric")`, `colab/binning.ipynb`) spreads it over ~7 bins.
The rollouts notebook still uses uniform bins.
