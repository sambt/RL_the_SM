# sm_rl — RL rediscovery of the Standard Model quark sector

An RL agent searches over **gauge groups + matter content**, computes the
gauge-singlet bound-state (hadron) spectrum with GroupMath, and is rewarded for
matching the observed spectrum. Target: rediscover **SU(3) colour with (u, d, s)
quarks** (antiquarks implied by CPT), reproducing the light-hadron spectrum.

The physics engine is the one from `GT Invariants.ipynb` / `spinSinglets.nb`
(GroupMath's `PermutationSymmetryOfInvariants`, wrapped as `GetSpins`).

## Layout

```
sm_rl/
  config.py            all knobs (Engine/Group/Env/Reward configs)
  physics/
    engine.py          Engine ABC + WolframEngine (cached GroupMath wrapper)
    groups.py          GroupLadder: type/rank stepping + rep dimension-index + clamp
    spectrum.py        Model -> hadron spectrum (CPT expansion, gcd normalization, caching)
  env/
    state.py           Model / Parton (ordered list, unique flavour tags)
    actions.py         flat maskable action space (+ optional cursor)
    sm_env.py          SMModelEnv: per-step reward, terminal bonus
  reward/
    base.py            RewardMetric interface (pluggable) + make_metric
    count.py           CountReward  (missing/extra by quantum numbers)   [default]
    hungarian.py       HungarianReward (optimal partial matching, graded)  [option]
    target.py          build the (u,d,s) target from the engine
  models/
    tokenizer.py       featurize group + partons (rep invariants, positional, edit flag)
    transformer.py     ActorCritic over [CLS, GROUP, P_1..P_n]
  algos/
    base.py            Algorithm ABC + collect_rollout + GAE   (algorithm-agnostic)
    ppo.py             PPO with invalid-action masking          [default learner]
scripts/    engine_smoke, phase0_validate, env_smoke, train_ppo
tests/      mock_engine + test_core (Mathematica-free)
```

## Setup

Needs a Wolfram kernel + GroupMath (the `Mathematica/` folder) and Python 3.10+.

### On a cluster (uv)

```bash
uv sync                       # core: numpy, scipy, wolframclient  (physics/env/reward)
uv sync --extra rl            # + torch, for training (choose your CUDA/CPU build)
uv run python scripts/doctor.py   # <-- run this FIRST; verifies kernel + files + a live eval
```

The Wolfram kernel is auto-discovered: `$WOLFRAM_KERNEL`, else a kernel binary on
`PATH` (`WolframKernel` / `wolfram` / `math`), else the macOS app path. GroupMath
files are taken from `$WOLFRAM_MATH_DIR` (else the repo's `Mathematica/`). So on a
cluster where `wolfram` is on PATH and you keep the `Mathematica/` folder, it just
works; otherwise:

```bash
export WOLFRAM_KERNEL=/path/to/WolframKernel      # or `which wolfram`
export WOLFRAM_MATH_DIR=/path/to/Mathematica      # holds GroupMath.m + spinSinglets.m
```

Everything is overridable in code via `Config().engine` (`EngineConfig`).

### Smoke tests / entry points

```bash
uv run python scripts/doctor.py            # preflight (kernel, files, live eval)
uv run python scripts/engine_smoke.py      # engine connects, group theory correct
uv run python scripts/phase0_validate.py   # reward makes the true model the argmax
uv run python scripts/env_smoke.py         # env loop + scripted (u,d,s) oracle
uv run python scripts/trial_ppo.py 3 48    # short honest PPO trial (starts from SU(2))
uv run python scripts/train.py --iters 200 --ckpt-every 10   # real run (logs + checkpoints + cache)
uv run python tests/test_core.py           # pure-Python unit tests (no Mathematica)
```

(A conda env with the same packages works too — e.g. `~/miniforge3/envs/jax/bin/python <script>`.)

## Design decisions (from the addendum)

- **Gauge group is an in-episode action.** `CHANGE_TYPE` (SU↔SO↔Sp↔exceptional,
  mapping to nearest Lie rank) and `CHANGE_RANK` (±1 within a family). Reps are
  addressed by **dimension-ordered index** (r0 trivial, r1 fundamental, …), and a
  group change **clamps** the index down to the new group's largest admissible
  rep (`GroupConfig.max_rep_dim`, default 248) — e.g. r3 of SU(3) → r1 of E8.
- **Identical partons stay distinct** — every parton carries a unique flavour tag
  (the "dummy quantum number"); look-alikes are never merged.
- **CPT**: each matter parton implies its conjugate (dual rep, negated charges)
  when the spectrum is computed, so `(u,d,s)` alone yields mesons + antibaryons.
  Toggle with `EnvConfig.cpt_auto_conjugate`.
- **Charge normalization** = per-column gcd of the computed hadron charges
  (`RewardConfig.normalization="gcd"`, the notebook's rule). Data-dependent; the
  Hungarian metric is the more robust alternative.
- **Positional encoding** on parton tokens (ordering is bookkeeping so the policy
  can track the most-recently-added parton, the default edit target).
- **Reward every step** (`EnvConfig.reward_mode="delta"` potential-based, or
  `"absolute"`), plus a `terminal_bonus` on an exact-match STOP.
- **Movable cursor** available (`EnvConfig.use_cursor=True`) but **off by default**.

## Runs: logging, checkpoints, resume, persistent cache

`scripts/train.py` writes everything to a run directory (`runs/<name>_<ts>/`):

```
runs/ppo_.../
  log.txt          timestamped human log (mirrors stdout)
  metrics.csv      per-iteration scalars (reward, entropy, losses, cache hit-rate, ...)
  config.json      full run configuration
  best_model.json  best model found so far (group + partons), updated on improvement
  cache.pkl        persistent spectrum cache (survives restarts; shareable across workers)
  checkpoints/     ckpt_NNNNNN.pt, latest.pt, final.pt  (policy + optimizer + RNG + iter)
```

Resume an interrupted job (continues iters, reloads the cache, appends to logs):

```bash
uv run python scripts/train.py --run-dir runs/ppo_20260709_101500 --resume latest --iters 100
```

- **Checkpoints** (`--ckpt-every K`) save policy + optimizer + RNG state + `iters_done`,
  so a resumed run is numerically continuous.
- **The spectrum cache** (`SpectrumCache`) is memoized on the canonical model key and
  persisted to `cache.pkl`; it auto-flushes every N new spectra and **merges on flush**,
  so several data-parallel workers can point `--cache` at one shared file and pool their
  (expensive) Mathematica evaluations. Hit-rate is logged each iteration.

## Swapping pieces

- **Reward metric** — `Config().reward.metric = "count" | "hungarian"`.
- **RL algorithm** — the env, tokenizer, and `ActorCritic` are algorithm-agnostic.
  Subclass `algos.base.Algorithm`, reuse `collect_rollout`, write a new update
  rule (e.g. GFlowNet trajectory-balance), and keep everything else. `PPO` is one
  such subclass.

## Status

Verified against the real engine: engine primitives, `(u,d,s)` target (54 states),
CPT ≡ explicit conjugates, reward argmax at the true model, scripted oracle
reaches an exact match, and the PPO loop trains without NaNs. Cluster-ready:
kernel auto-discovery, `doctor.py` preflight, file logging, checkpoint/resume, and
a persistent+shareable spectrum cache. Next: longer training runs, an evolution
baseline, and (optionally) a GFlowNet learner.
