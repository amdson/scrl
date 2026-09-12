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
<https://colab.research.google.com/github/amdson/sillyrl/blob/colab-sweep/colab/sweep.ipynb>.
For a private repo, add a Colab secret `GITHUB_TOKEN` with read access.

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
run.py         CLI
```
