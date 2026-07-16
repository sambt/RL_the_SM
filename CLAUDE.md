# CLAUDE.md

Guidance for working in this repo. See `README.md` for user-facing setup/usage.

## What this is

`sm_rl` — an RL framework that searches over **gauge group + matter content** to
rediscover the Standard Model quark sector. An agent builds a model
(a gauge group + ordered list of matter "partons"), a Mathematica/GroupMath
engine computes its gauge-singlet bound-state (hadron) spectrum, and the reward
measures agreement with a target spectrum. Target: **SU(3) colour with (u,d,s)
quarks** reproducing the light-hadron spectrum (antiquarks come from CPT).

The physics is a port of `GT Invariants.ipynb` / `spinSinglets.nb` (GroupMath's
`PermutationSymmetryOfInvariants`, wrapped as `GetSpins`).

## Architecture — strict layering, each layer decoupled

```
config.py              all knobs: Engine/Group/Env/Reward dataclasses (Config aggregates them)
physics/
  engine.py            Engine ABC + WolframEngine (cached GroupMath wrapper); Rep = Dynkin-label tuple
  groups.py            GroupLadder: SU/SO/SP/EX families, type/rank stepping, dim-ordered rep index + clamp
  spectrum.py          Model -> spectrum: resolve_fields (CPT), compute_spectrum, canonical_key, normalize
  cache.py             SpectrumCache: dict subclass, persistent pickle, auto-flush + merge across workers
env/
  state.py             Model / Parton (ordered list; each parton has a unique `flavour` tag)
  actions.py           ActionSpace: flat maskable actions (STOP, TYPE/RANK +/-, ADD, REMOVE, SPIN, REP, CHARGE)
  sm_env.py            SMModelEnv: gym-like reset/step, per-step reward, terminal bonus
reward/
  base.py              RewardMetric ABC + make_metric + MatchResult
  count.py             CountReward (missing/extra by quantum-number signature)   [default in code]
  hungarian.py         HungarianReward (optimal partial matching, graded)  [BROKEN as a learning signal -- see gotchas]
  f1.py                F1Reward (quality-weighted F1, bounded [0,1], 1==exact)    [train.py default]
  target.py            build_target: constructs (u,d,s) SU(3) model + its spectrum
models/
  tokenizer.py         featurize obs -> tensors (group-invariant rep scalars: log dim, Dynkin, Casimir, N-ality)
  transformer.py       ActorCritic transformer over [CLS, GROUP, P_1..P_n]; masked action head + value head
algos/
  base.py              Algorithm ABC + collect_rollout + compute_gae (algorithm-agnostic)
  ppo.py               PPO w/ invalid-action masking + checkpoint save/load  [default learner]
utils/logging.py       Logger: log.txt + metrics.csv + config/best_model json + ckpt dir
scripts/               doctor, engine_smoke, phase0_validate, env_smoke, trial_ppo, train_ppo, train
tests/                 mock_engine.py (Mathematica-free Engine) + test_core.py (pure-Python unit tests)
```

**Data flow per step:** `SMModelEnv.step(action)` → `ActionSpace.apply` (pure
Python, no engine call) → `spectrum_of` (engine + `SpectrumCache`, memoized on
`canonical_key`) → `RewardMetric.match` → scalar reward. The env emits a
structured obs `{"model", "mask"}`; the tokenizer/policy turn that into tensors.

## Key design invariants (don't break these)

- **Engine is the only thing that touches Wolfram.** Nothing outside
  `physics/engine.py` imports `wolframclient`. Everything else runs against
  `tests/mock_engine.py`. All engine methods are memoized (results are pure
  functions of args); `get_spins` is the expensive call.
- **Reps are addressed by dimension-ordered index** (0=trivial, 1=fundamental),
  not raw Dynkin labels. A group change **clamps** the index down to the new
  group's admissible reps (`GroupConfig.max_rep_dim`, default 248).
- **Partons stay distinct**: each carries a unique `flavour` id; look-alikes are
  never merged.
- **CPT** (`EnvConfig.cpt_auto_conjugate`): each parton implies its conjugate
  (dual rep, negated charges) at spectrum time, so (u,d,s) yields mesons +
  antibaryons.
- **Spectrum caching is on `canonical_key`** (order-invariant over resolved
  fields). Target and prediction must be normalized by the *same* rule
  (`RewardConfig.normalization`, default per-column gcd — data-dependent).
- **Charges** use the notebook's scaled-integer convention `[EM, B, S, I3]`
  (EM/B/S = physical×3 integers, I3 in halves); a spectrum row is
  `[charges..., su2_spin_label]` where the SU(2) label is 2S+1.
- **The RL algorithm is swappable.** env, tokenizer, and `ActorCritic` are
  algorithm-agnostic. New learner = subclass `algos.base.Algorithm`, reuse
  `collect_rollout`; touch nothing else.

## Running things

```bash
uv sync && uv sync --extra rl          # core, then torch for training
uv run python scripts/doctor.py        # RUN FIRST: verifies kernel + files + a live eval
uv run python tests/test_core.py       # pure-Python tests, no Mathematica
uv run python scripts/train.py --iters 200 --ckpt-every 10   # real run
```

- Kernel auto-discovered via `$WOLFRAM_KERNEL` → PATH → macOS app path; GroupMath
  from `$WOLFRAM_MATH_DIR` → repo `Mathematica/`. Overridable via `EngineConfig`.
- `train.py` writes to `runs/<name>_<ts>/` (log.txt, metrics.csv, config.json,
  best_model.json, cache.pkl, checkpoints/). Resume with
  `--run-dir <dir> --resume latest`. Cache is shareable across data-parallel
  workers (merge-on-flush).
- Prefer `mock_engine` for any test/dev that doesn't need real group theory —
  it's fast and needs no Wolfram kernel.

## Notes / gotchas

- **Do not use `hungarian` + `normalization="none"` as a learning signal.** Its
  spurious penalty (`lam_spur` per unmatched prediction) is additive and
  unbounded while the number of states it can match saturates, so the score is a
  decreasing linear function of how many hadrons a model predicts
  (`corr(n_pred, score) = -0.995` measured over cached SU(3) models). The
  reward-maximising model therefore predicts *nothing*, and SU(3) -- the only
  group forming many singlets at `d_max=3` -- ranked **27/30** among groups: the
  gradient pointed away from the SM. Two 5k-iteration runs (`runs/tier1_*`)
  burned 24h each confirming this; the free-group one never beat the score its
  *randomly initialised* policy got at iteration 0. Use **`f1` + `physical`**
  (the `train.py` defaults): SU(3) rises to 6/30, `corr` becomes +0.27, and 100%
  of SU(3) models beat the empty model (was 17%).
- **`normalization` is load-bearing, not cosmetic.** Charges are stored as
  physical x3, so the target's charge quantum is **3**. With `"none"`, a
  one-unit charge error costs 3.0 >= the match radius -> zero partial credit ->
  no gradient toward the right charges. `"physical"` divides by
  `RewardConfig.charge_scale` (deterministic, unlike `gcd`) so the quantum is 1.
  Changing `normalization` also **rescales the score**, so scores are not
  comparable across runs with different settings (the `cluster` run's 0.333
  under `gcd` is 0.136 under `none` -- the same model).
- `scripts/train.py` defaults reward metric to **f1**; the `RewardConfig`
  dataclass default is **count**. Metric is auto-reused from a run's
  `config.json` on resume.
- Reward changes are cheap to validate offline: `canonical_key` is pure physics,
  so the ~156k cached spectra stay valid across reward/normalization changes.
  Score them against the real target directly instead of burning cluster time.
- Episodes start at SU(2) by default (`ActionSpace`/ladder default), but
  `PPOConfig.start_group` and `train.py --start-group` default to `SU3`.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set on import to dodge an OpenMP double-load
  abort on conda/macOS.
- `scripts/train_ppo.py` and `scripts/trial_ppo.py` are older/short entry
  points; `scripts/train.py` is the full-featured runner.
