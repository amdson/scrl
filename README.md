# maze_consistency

Prototyping reward-conditioned self-consistency (`plan.md`) on one fixed maze, in JAX.

Current state is the data side only: a hand-edited 12x12 maze, 100k deterministic random-walk rollouts,
the token format, the model definition, and exact ground truth for checking. The training algorithm is
being redesigned. The first full implementation of `plan.md` (training loop, losses, FQI baselines,
E1–E5 experiments) is in git history at commit `f845a73`, and `NOTES.md` is its write-up.

## Commands

```
uv sync
python run.py dataset      # build data/canonical/ from data/canonical/maze.txt
python run.py truth        # exact random-walk ground truth figure
python run.py show [i]     # print a rollout on the maze and as tokens
python tests/test_basic.py
```

## Colab

`colab/sweep.ipynb` clones this repo, rebuilds the dataset and the exact test set, and runs the
loss-configuration sweep, saving runs to Google Drive so an interrupted session resumes where it stopped.
Open it in Colab with File → Open notebook → GitHub, or at
<https://colab.research.google.com/github/amdson/scrl/blob/main/colab/sweep.ipynb>.
For a private repo, add a Colab secret `GITHUB_TOKEN` with read access.

## Interval consistency losses

`consistency_losses.md` is the note; `maze_consistency/consistency.py` implements it and
`consistency_demo.ipynb` is a short run showing what the objectives look like on real rollouts (magnitude,
spread across trajectories, scaling with length, and gradient pull relative to the data loss).

Two teacher-forced passes over the same rollout -- MODE = NOR and MODE = the rollout's own bin -- give
`u_t`, `v_t` and `b_t = log q_t(R)` at every env-step boundary, and `delta_t = v_t - u_t + b_{t-1} - b_t`.
Losses over all n(n+1)/2 intervals: `local`, `all` (the 2(n+1)/n * Var(c) shortcut), `all_scaled`, `mixed`,
`poly_len`, `multiscale`. No model change was needed -- the value head is already read at every state slot.
`python tests/test_consistency.py` checks the fast forms against explicit interval enumeration in both
values and gradients.

Training with one: `LossConfig(mc=True, cons=True, cons_loss="all_scaled", w_cons=0.14)`. The term runs on its
own batch of `cons_batch` rollouts tokenized in both modes and is added into the main gradient, so the data
loss is untouched and a consistency run differs from its baseline by exactly that one term. `cond_gap` and
`info_gain` are logged beside it as the collapse check -- every one of these objectives is minimized by
ignoring R entirely (`v == u`, `b` flat in `t`), so a falling consistency loss is only good news while those
two hold up.

```
python run.py cons-sweep 3000   # mc baseline vs every consistency loss -> runs/cons/
python run.py cons-plot         # test curves + collapse diagnostics
```

`evaluate.conditioning_response` asks whether the MODE token moves the policy at all: one forward pass per
(start cell, MODE), no rollouts, so it avoids the compounding that makes `return_sweep` unreadable at T = 200.
It reports `P(step toward the goal)` against the exact `piR*` and a **captured fraction**,
`1 - KL(piR* || model) / KL(piR* || uniform)` -- the share of the conditioning signal learned, where the
denominator is what a MODE-ignoring model scores. Most of a raw `act_kl` is the irreducible entropy of `piR*`,
so the fraction is the readable version. All three Colab notebooks plot it.

`colab/consistency_sweep.ipynb` is the same comparison on a GPU, writing to Drive:
<https://colab.research.google.com/github/amdson/scrl/blob/main/colab/consistency_sweep.ipynb>.
It defines the sweep, plots and table as plain local functions so every parameter is editable in the Colab
session; `experiments.py` is the packaged equivalent behind the CLI. Each objective gets its own
`lambda_cons` chosen for roughly equal gradient pull, since raw `all` and `poly_len` sit ~100x above the
scaled family in loss value. `consistency.exact_terms` gives the same quantities for the true model from the
DP, where every loss is 0, so the sweep's held-out `cons/` curves are absolute.

## Model rollouts as training data

`augment.py` continues real prefixes with the model, asking for more reward than the original achieved, and
feeds the spliced trajectories back into training. It targets the coverage gap directly: of 66,609 far-start
random walks, none achieved their start's best outcome, so there was nothing to imitate.

- Continuations step the real maze (`maze.next_open`), not the dynamics head, so every spliced trajectory is
  a genuine environment trajectory under a behaviour policy that switches at `tau`.
- R is relabelled to the bin achieved, never the bin requested.
- Every head trains on the mixture (`train(mixer=RolloutBuffer(...), mix_frac=...)`). The interval identity is
  about one joint distribution; rollouts in the R head alone would make the heads disagree about it.
- The DP ground truth describes the random walk, so `act_kl` / `value_kl` for rollout-trained runs are
  references for a different distribution. `RolloutBuffer.history` is the readout that stays valid: asked for
  a higher reward from a real prefix, how often does the model get one, against the exact random-walk rate
  from the same prefixes.
- `td` is refused with a mixer: its importance weight assumes the uniform behaviour policy.

`colab/rollouts.ipynb` runs `{mc, mc_all} x {data only, + rollouts}`.

## Outcome binning

`Maze(binning=...)` chooses how arrival time L is partitioned into the K value-head bins.

`"uniform"` (the default, unchanged) splits L evenly -- equivalently log R evenly, since log R = L log gamma.
On this maze that wastes almost every bin: T = 200 makes each bin ~18 steps wide while the longest
shortest-path is 20, so **all optimal play lands in bins 10-11** and bins 1-9 all mean "wandered for 37-200
steps". Conditioning on them requests near-identical behaviour (exact P(step toward goal) moves only
0.30 -> 0.41 across bins 0..9), so a model that ignores MODE is nearly correct on 92% of its data.

`"geometric"` splits log L, concentrating resolution where optimal play lives. Bins usable by optimal play:

| K | uniform | geometric | exact KL(optimal \|\| piR*) at each start's best bin, uniform -> geometric |
| --- | --- | --- | --- |
| 6 | 1 of 5 | 3 of 5 | 0.747 -> 0.355 |
| 12 | 2 of 11 | 7 of 11 | 0.391 -> 0.163 |
| 24 | 3 of 23 | 11 of 23 | 0.274 -> 0.077 |

Only the labels change, so the stored rollouts are untouched: a variant needs no dataset rebuild, just its
own exact test set. `Maze.empty_bins` reports bins no integer arrival time can land in (geometric leaves
18, 21, 22 empty at K = 24) -- dead value-head classes with h = 0 everywhere, which `build_testset` now skips.
`colab/binning.ipynb` sweeps the grid.

## Maze and returns

`data/canonical/maze.txt` is the layout, hand-edited; the build reads it and never overwrites it.
There is no fixed start: each random walk starts at a uniformly random open cell, runs up to T = 200 steps,
and stops at the goal. Reward is 1 on reaching the goal and 0 otherwise, discounted by gamma = 0.95: a
rollout arriving in L steps has total return `R = gamma**L`, counted from its own start, and one that never
arrives has `R = 0`. The optimal return from a start s is `gamma**dist(s)`. The value head predicts R over 12 bins: bin 0 is `R = 0`, and bins 1..11
split `log R` evenly between `log(gamma**T)` and 0.

## Token format

Each sequence slot is a triple `(kind, x, y)`. Actions, return bins, NOR and PAD carry null coordinates;
positions carry their column and row. The model embeds a slot as `E_kind + E_x + E_y`, so a cell costs
`W + H` embedding rows instead of `W * H`. A sequence is `[MODE] pos_0 a_1 pos_1 ... a_L pos_L PAD ...`,
length `2 + 2T`, where MODE is NOR or a return bin and pos_0 is the start cell.

## Model

One causal transformer with two heads. The next-token head is a single softmax over everything that can
follow a slot: the 4 actions, the cells, and END (the padding after reaching the goal). At a state slot it
is the policy, pi in NOR mode and pi_R in R mode; at an action slot it is the dynamics model
P(s' | s, a). It trains by teacher forcing: one cross-entropy on the sequence shifted by one slot
(`Tokenizer.next_targets`, `model.next_token_loss`). The value head is a categorical over the 12 return
bins, read at state slots.

## Layout

```
maze_consistency/
  env.py       maze, dynamics, return and bin definitions, random walks
  dataset.py   canonical dataset build, stats, figure
  dp.py        exact random-walk ground truth
  tokens.py    slot format and rendering
  model.py     causal transformer: next-token head and value head, teacher-forcing loss
  viz.py       maze plotting
  train.py     training loop: LossConfig (next-token + optional MC / TD / A), periodic eval hook
  evaluate.py  mode accuracy and in-maze return sweeps
  testset.py   exact test set from the DP, and scoring against it
  experiments.py  loss-configuration sweeps with test tracking, and their plots
  consistency.py  interval consistency losses (consistency_losses.md), plus brute-force references
run.py         CLI
```
