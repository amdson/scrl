# maze_consistency — reward-conditioned self-consistency on a fixed maze (JAX)

Implementation of `plan.md`: one causal transformer with a next-action head (`pi`, NOR mode),
a categorical return head (`V_t`, NOR mode) and a return-conditioned action head (`pi_R`, R mode),
trained on random-walk data with the consistency identities

```
(A)  pi_R(a|s,R) V_t(R|s) = pi(a|s) V_{t+1}(R|s,a)
(B)  V_t(R|s)             = sum_a pi(a|s) V_{t+1}(R|s,a)
```

See `NOTES.md` for results, pitfalls and deviations from the plan.

## Setup

```
uv sync                                  # jax / flax / optax / numpy / matplotlib (CPU)
.venv/bin/python tests/test_basic.py     # DP identities, tokenizer, one train step, eval
```

## Running

```
python run.py truth                # DP ground-truth figures -> results/truth/
python run.py E1 --profile cpu     # H1  MC vs TD value error, N sweep      -> results/E1/
python run.py E2                   # H2  MC / TD / TD+A at R = -d*           -> results/E2/
python run.py E2b                  # H2 follow-up: --rtg bins, explicit --children, rare-but-observed
python run.py E3                   # H3  locked-door maze, decode rules      -> results/E3/
python run.py E4                   # H4  TD+A vs neural + tabular FQI        -> results/E4/
python run.py E4 --drop_actions 0.3   #    with coverage gaps (plan §9)
python run.py E5                   # H5  loop closures                       -> results/E5/
python run.py plot E2              # regenerate a figure from saved metrics
python run.py report results/E2/TDA_N10000_s0     # per-run report + pi_R arrows
python run.py single --loss TDA --N 10000 --steps 2000 --name mytest [--maze door] [--rtg] [--children]
```

Profiles (`maze_consistency/experiments.py::PROFILES`): `cpu` (d=64, 2 layers, batch 64, 2k steps,
~50 ms/step on an 8-core laptop), `gpu` (the plan's d=128, 4 layers, batch 256, 20k steps), `quick`.
Finished runs are skipped; `--force` reruns.

Two flags added after the first results (see `NOTES.md` §2 E2b): `rtg` bins the value head as
`[FAIL, return-to-go −T..0]` so terminal facts are t-independent, and `children` adds the plan's explicit
4-child (A)/(B) terms with the terminal applied analytically. Both are off by default to match the plan.

## Layout

```
maze_consistency/
  env.py          maze generation (open room / backtracker / parametric door maze), dynamics, random walks
  dp.py           ground truth: h[t,s,R], child values, Doob h-transform pi_R*, Q_rw, Q_opt, occupancy
  tokens.py       sequence layout, state-token index map, r(R)
  model.py        flax transformer + 3 heads
  losses.py       L_pi, L_piR, L_MC, L_TD, L_A, L_term, closures, neural-FQI loss
  train.py        TrainConfig, jit'd step, loop with closures, checkpoints, metrics.json
  fqi.py          tabular FQI, biased-data helper
  eval.py         metrics vs DP, model-as-policy rollouts (pi_R / posterior tilt / EV tilt / FQI)
  viz.py          maze/arrow/heatmap rendering, curves, per-run report
  experiments.py  run_e1..run_e5, plot_e1..plot_e5, PROFILES
run.py            CLI
tests/            sanity tests
```

## Metrics (per eval, saved in `metrics.json`)

`value_err[t]` TV(V_t, h_t) · `er_err[t]` |E_V[R] − E_h[R]| · `logp_rstar_err[t]` |log V_t(−d*) − log h_t(−d*)| ·
`piR_err[t]` KL(pi_R* ‖ pi_R) at R∼r(R) · `consistency[t]` (A) residual at data actions ·
`opt_err` KL(pi_R* ‖ pi_R) at R=−d* on shortest paths · `rollouts{policy}` solve/optimal rate, mean steps,
door_choice · `q_err[t]`, `overest[t]` (FQI).
