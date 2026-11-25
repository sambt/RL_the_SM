# Experiment log — rediscovering the SM quark sector with RL

Run-by-run narrative of how the agent has (and hasn't) learned to rebuild
**SU(3) colour with (u, d, s)**. Each phase records *what we did*, *the outcome*,
and *how it informed the next step*. Terse mechanism notes live in
`CLAUDE.md` (gotchas) and `docs/reward_metrics.md` (the metric math); this file
is the chronology.

**Goal.** An agent builds a gauge group + ordered matter content; the
Mathematica/GroupMath engine computes the gauge-singlet (hadron) spectrum; the
reward measures agreement with the target SM light-hadron spectrum. An *exact*
match means the predicted spectrum equals the SM's.

**Status (2026-09-02, evening): still open; optimizer tuning exhausted.** Best
honest result remains a single sampled model with the SM *structure* (SU(3),
three fundamental fermions) but wrong U(1) charges (`f1nostop`, 0.786). A
diagnostic proved the remaining gap is an **exploration failure, not a
reward-landscape trap**. Every optimizer-side lever has now been tried and
failed: novelty bonus large (`f1novel`) and small (`f1novel2`/`f1novelws`) — in a
space this big "first visit" is every step so the bonus is just a diversity
force that pins entropy near uniform; and a gentle warm start (`f1ws`, seed at its
converged entropy, novelty off) — sits on a 0.72 plateau for 5000 iters, visiting
only 4.8k new configs in 5M steps. PPO either never concentrates or never leaves.
Next: a **structural** change — set-charge actions (make the valley crossing 1–2
moves) + self-imitation to lock in rare good samples, or MCTS / Go-Explore on the
deterministic simulator.

## Run scoreboard

| Run | Key change | Best score | Found SM? | Notes |
|-----|-----------|-----------:|:---------:|-------|
| `tier1_*` | hungarian + `none` norm | — | no | Broken learning signal (see Phase 1); 2×24h burned |
| `f1fix_su3` | f1 + physical, **start SU3** | **1.000** @551 | **yes**, repeatedly | Proof the objective is learnable from inside the basin |
| `f1fix` | same, **start SU2** | 0.726 @351 | no (0 exact ever) | Local-optimum trap; motivated exploration work |
| `f1rand` | random start group | 0.667 | no (0 exact) | Uniform-over-16 too diffuse |
| `f1curr` | curriculum SU3→random | **1.000** @994 | **yes** | Works, but *hands over* the group — biased |
| `f1nostop` | no-STOP fixed horizon | 0.786 @1220 | no (0 exact) | Right structure, wrong charges; entropy stayed high |
| `f1novel` | novelty bonus (β=0.05) + opt bundle | 0.652 @139 | no (0 exact) | Novelty dominated (~70% of return) → wandered, stalled flat |
| `f1novel2` | retuned novelty (β=0.01, first-visit, annealed), fresh | 0.712 @126 | no (0 exact) | 458 iters/24h; on par with f1nostop at equal iters; entropy pinned ~2.4; intrinsic still ~40% of return |
| `f1novelws` | same config, **warm-started** from f1nostop 0.786 policy | 0.726 @129 | no (0 exact) | Entropy 0.5→2.2 in ~250 iters (ec restarted at 0.03 + novelty); never re-found 0.786, lost the structure |
| `f1ws` | warm start done right: seed f1nostop ckpt_4500, ec 0.01 fixed, novelty off | 0.722 @642 | no (0 exact) | 5000 iters in 6h (99.9% cache hits); best flat from iter 0; entropy 0.4–0.7; only 4.8k new configs in 5.1M steps. **Plateau policy never leaves its neighbourhood** |
| `f1set` | **action space**: set-charge (64 actions) + cursor, fresh, no novelty | 0.665 @200 | no (0 exact) | 5000 iters in 4.7h. **Entropy-farming attractor**: SU(N≥5)+[singlet,singlet,fund] fermions, then rank-up so the fund is inert and 71% of steps are free-entropy charge sets on it (H=3.5 there vs 0.9 on real steps). Flat 0.549 |

---

## Phase 0 — Initial framework

**What.** Built the layered framework (`config → physics → env → reward → models
→ algos`): a maskable action space that edits a model (group ladder + partons
with rep/spin/charges), a cached Wolfram engine for singlet spectra, PPO with
invalid-action masking, and CPT auto-conjugation so (u,d,s) alone yields mesons
+ antibaryons. Reward keyed on an order-invariant `canonical_key` (per-physics),
which is also the spectrum cache key.

**Outcome.** End-to-end pipeline worked; the question became *which reward* makes
the SM the reward-maximiser.

**Informed next.** Need a reward whose optimum actually *is* the SM.

## Phase 1 — Fixing the objective (count/hungarian → f1 + physical)

**What.** Early runs (`tier1_*`) used the Hungarian metric with
`normalization="none"`. Replaced it with **F1 (bounded [0,1], 1==exact)** and
**`physical` normalization** (divide charges by the ×3 scaled-integer quantum so
a one-unit charge error is 1.0, inside the match radius).

**Outcome.** The Hungarian+none signal was *actively misleading*: its unbounded
spurious penalty made score a decreasing function of how many hadrons a model
predicts (`corr(n_pred, score) = −0.995`), so the reward-maximiser predicted
*nothing* and SU(3) ranked **27/30** among groups — the gradient pointed away
from the SM. Two 5k-iter runs burned 24h each confirming it. Under **f1 +
physical**, SU(3) rose to **6/30**, `corr` became **+0.27**, and 100% of SU(3)
models beat the empty model (was 17%).

**Informed next.** With a learnable objective, test whether PPO can actually
climb it — and from where.

## Phase 2 — Discovering the exploration problem (`f1fix`)

**What.** Two identical f1+physical runs differing only in start group: SU(3)
vs SU(2).

**Outcome.** **Start matters enormously.** SU(3)-start hit **1.000 at iter 551**
and kept re-finding the SM (172k exact terminal hits). SU(2)-start plateaued at
**0.726 and never once matched** across ~4.6M episodes — a hard, flat local
optimum (trivial-rep parton + wrong baryon-number column), structurally distant
from the SM basin, with entropy collapsed before it could cross.

**Informed next.** It's a genuine local-optimum / premature-convergence problem,
not slow progress. A *structural* change to exploration is needed — more
iterations alone won't help.

## Phase 3 — Adding exploration I: randomized starts + softened terminal bonus

**What.** (a) Per-episode start-group scheduling: `random` (uniform over a
curated 16-group low-rank pool) and `curriculum` (start SU(3), anneal
p(base) 1→0 over 2000 iters). (b) Softened the terminal reward from the legacy
`exact`-only +10 cliff to `graded` (`terminal_bonus × final_score` for any STOP),
so partial models get partial terminal credit.

**Outcome.** **Split result.** `f1curr` reached **1.000 at iter 994** and held it
through the anneal — the trap was broken. But `f1rand` never matched (stuck
0.643→0.667, G2/2-parton basin): uniform-over-16 starts put only ~6% of episodes
in SU(3), too diffuse to build consistent gradient. Crucially, **even f1rand's
6% SU(3) starts weren't enough** — proving the bottleneck isn't only *reaching*
SU(3) but getting *concentrated* experience once there.

**Informed next.** Curriculum "works" but is unsatisfying — it *tells* the agent
the group is SU(3) rather than letting it discover it, and that bias leaks into
the policy. We want discovery without the hint. Hypothesis: the early-**STOP**
escape hatch lets the policy commit to a mediocre basin before exploring — remove
it.

## Phase 4 — Adding exploration II: no-STOP fixed horizon (`f1nostop`)

**What.** Added `allow_stop=False`: STOP is masked entirely; an episode ends
*only* at `max_steps` or on an **exact match** (absorbing goal). This is the
fixed-horizon setup common in the discovery-RL literature — it forces the policy
to keep editing rather than bail early. Random start, dense delta reward.

**Outcome.** **The most encouraging run, and the most informative.** Entropy
**never collapsed** (held ~2.0–2.6 vs f1rand's ~0.05) — removing STOP did sustain
exploration. Best model climbed to **0.786** and is **structurally the SM**:
SU(3), **three fundamentals, all spin-½ fermions** — only the U(1) charges are
wrong. But it plateaued there (0 exact), and it's ~23× slower (128-step episodes
over novel models → low cache hit rate). It timed out at iter 923/5000 on a 24h
wall; resumed on `intermediate` to 5000 — **still 0.786**.

**Informed next.** The problem is no longer group discovery — it's **U(1) charge
fine-tuning in a discrete lattice**. Before spending more compute, ask the
decisive question offline: is 0.786 a *reward-landscape local optimum* (walls) or
an *exploration failure* (a path exists, PPO didn't find it)? These need opposite
fixes.

## Phase 5 — The decisive diagnostic (`scripts/charge_landscape.py`)

**What.** Offline, using the same physics + metric: (A) best-first search over
the ±1 charge lattice around the plateau, and (B) a direct lattice walk from the
plateau charges toward the known u,d,s target. Ran against the warm shared cache
(no new cluster training).

**Outcome.** **Exploration failure, confirmed both ways.** (A) reached an
**exact match within 1500 evals**, and the plateau is **not** a strict local max.
(B) reached exact in 17 steps but **through a shallow ~0.08-deep valley** (scores
dip 0.786→0.71 before climbing to 1.0). Bonus finding: the exact solution (A)
found has *different* charges than the canonical target — **multiple exact charge
assignments exist** (a physical degeneracy), so the target set is larger than one
point. The valley explains the plateau precisely: a collapsed-/low-entropy policy
won't pay the temporary score cost to cross it, so it pins on the near side.

**Informed next.** Optimization knobs *should* work (the landscape connects), but
they must specifically (i) keep entropy alive to cross the valley and (ii) direct
exploration into charge space rather than wandering the whole action set.

## Phase 6 — Current: novelty bonus + optimization bundle (`f1novel`)

**What.** Added a **count-based novelty (intrinsic) reward**
(`novelty_coef`, default off): each step adds `beta / sqrt(n[canonical_key])`,
keyed on physics. Near the plateau, "new physics" == new charge configs, and it
self-anneals (revisiting the plateau earns ~0). Folded into a bundle aimed at the
Phase-5 diagnosis:

- `--novelty-coef 0.05` — pay the cost of crossing the ~0.08 valley toward new charges.
- `--ent-coef-final 0.02` — stop entropy bleeding to 0.005 (the premature-collapse driver).
- `--max-steps 48` — ~42 episodes/2048-step update vs f1nostop's 8 → lower-variance gradients, less overfit.
- `--epochs 2` (from 4) — gentler PPO updates, slower entropy collapse.
- Still no-STOP + random start.

**Outcome.** **Failed — the bonus over-shot.** Across two 24h blocks (rollout
2048 then 1024) it reached only **best 0.652 and flatlined from iter ~176**, with
**0 exact** and entropy pinned high (~2.6, never settling). Root cause, from the
arithmetic: with delta reward the *extrinsic* return telescopes to ≈ final_score
≈ 0.6, but the novelty term pays `0.05/√n` on each of 48 steps with ~56% of steps
landing on new models, so *intrinsic* return ≈ 1.5 — **~70% of `ep_return ≈ 2.1`
was novelty, not task.** The agent was trained mostly to *find new physics* and
did exactly that: wandered the whole model space (all groups, near-trivial
charges) instead of honing SM charges. Two sub-flaws compounded it: novelty paid
*every* step (accumulates over the 48-step horizon) and `Σ1/√n` diverges (revisits
keep paying). Also confirmed: dropping rollout 2048→1024 ~halved cost to ~2.8
min/iter (throughput lever works).

**Informed next.** Rebalance novelty to *shape* exploration, not own it (Phase 7):
cut the coefficient (early magnitude), make it first-visit-only (kill per-step
accumulation + camping), and anneal it to 0 (clean late exploitation).

## Phase 7 — Retuned novelty (`f1novel2`)

**What.** Three complementary fixes, each targeting a distinct failure of Phase 6:
- `novelty_coef 0.05 → 0.01` — fixes the early-magnitude domination (target
  intrinsic ≲ 30% of return, not 70%). The only lever that controls magnitude.
- `novelty_mode = first_visit` (new) — pay the bonus **once** per config on its
  first-ever visit, 0 thereafter. Kills the per-step accumulation over the long
  horizon and the divergent-revisit / camping reward; makes intrinsic self-fade
  as coverage grows.
- `novelty_anneal_iters` (new) — linearly anneal the bonus → 0 over training
  (PPO scales `env._novelty_scale` by `1 − iters_done/anneal_iters`), so late
  training is pure exploitation and can sit exactly on an exact match.
Everything else unchanged (no-STOP, random start, max_steps 48, epochs 2,
ent-floor 0.02, rollout 1024).

**Outcome (both 24h blocks on `sapphire`, 2026-08-25→26). Failed.**

| | `f1novel2` (fresh) | `f1novelws` (warm from f1nostop `final.pt`) | `f1nostop` @ same iter |
|---|---|---|---|
| iters in 24h | 458 (~3.1 min/iter) | 575 (~2.5 min/iter) | — |
| best | 0.712 @126, flat after | 0.726 @129, flat after | 0.691 @450 (0.786 only @1138) |
| entropy | 2.4–2.6 throughout (≈uniform; max ≈2.8) | 0.52 @0 → 1.1 @50 → 1.7 @100 → 2.2 @275, then flat | 2.2 @450 |
| per-iter cache-miss frac | ≈0.60, *flat* over 450 iters | 0.13 @25 → 0.6 by iter 250 | — |
| exact | 0 | 0 | 0 |

Best models: f1novel2 `SU3[r1,r2,r1]` (one sextet); f1novelws `SU3[r0,r1,r2]`
(singlet+fund+sextet) — the warm start did not even keep the 3-fundamental
structure of its seed.

What went wrong:
1. **First-visit novelty does not self-fade here.** The premise was that
   first-visit bonuses vanish as the space is covered. But the reachable space
   (4 charge dims × 13 values per parton, 3+ partons, reps, spins, 16+ groups) is
   astronomically larger than the ~1.9M cached configs, so for a high-entropy
   policy ~60% of *every* step is a first visit (lower bound from per-iter cache
   misses against the shared cache). That is ≥0.29/episode of intrinsic reward at
   β=0.01 × 48 steps — comparable to the ~0.4 extrinsic a near-random policy earns.
   Intrinsic was ~40–45% of return, not ≲30%, and constant, not fading. The only
   fade is the 3000-iter anneal, i.e. ~6.5 days at this throughput.
2. **Novelty is a diversity force = an entropy bonus in disguise.** Actions that
   revisit (no-ops, backtracks, the policy's habitual path) earn 0; anything else
   earns β. That pushes mass *away* from whatever the policy does most, which is
   exactly what stops it concentrating. The warm start shows this directly:
   ep_return stayed ≈0.67–0.75 while entropy went 0.5→2.2 because intrinsic
   filled in what extrinsic lost (extrinsic ≈0.61→≈0.42, intrinsic ≈0.06→≈0.28).
   The return landscape between "sit on the plateau" and "wander" was made flat.
3. **The warm-start entropy schedule was wrong (my design error).** `--init-from`
   resets iters to 0, so `ent_coef` restarted at 0.03 — 5× the 0.006 the seed had
   converged under. In every run so far the policy sits at entropy ≈2.3–2.5 while
   ec ≳0.025 (f1nostop only concentrated once ec fell below ~0.025 at iter ~1000),
   so re-inflation to uniform was guaranteed even without novelty.
4. **The seed was weaker than advertised.** f1nostop's `final.pt` has mean
   episode return ≈0.62 (entropy 0.5–0.7, with a dip to 0.16/0.37 at iter 4800);
   0.786 was its single best *sample* at iter 1138, not what the policy reliably
   produces. Warm iter-0 best was 0.642.
5. **The fresh run is not evidence either way.** At iter 450 it matches f1nostop
   at iter 450 (best 0.71 vs 0.69, entropy 2.4 vs 2.2). f1nostop needed 1138
   iters for 0.786; 24h buys ~460. Continuing it to 3000+ iters (≈6.5 days) would
   only test whether "f1nostop plus a constant entropy-like bonus" beats f1nostop.

Instrumentation gap: intrinsic reward is not logged separately (`info["novelty"]`
is dropped by `collect_rollout`), so the shares above are bounds from cache-miss
counts. Any future intrinsic-reward run should log `novelty_return` per iter.

**Informed next.** Count/first-visit novelty over the raw config space is the
wrong tool for this problem; drop it (or restrict it to a *relative* notion, e.g.
novelty within the current group/structure). The cheap, informative next test is
the warm start done right: seed from f1nostop, keep `ent_coef` near what the seed
converged under (~0.01 fixed, or 0.01→0.005), novelty off, epochs 2, and see
whether a gently-perturbed plateau policy crosses the ~0.08 charge valley on its
own. If it does not, the Phase-5 result (exact match within 1500 best-first evals
of the plateau) argues for a hybrid: RL finds the structure, a local charge
search (or a charge-only sub-policy) finishes it.

Runs as an A/B pair (identical config, separate run dirs):
- `f1novel2` — **fresh** (from scratch). Tests "find the structure *and* cross".
- `f1novelws` — **warm-started** from f1nostop's `final.pt` (the 0.786 plateau
  policy, which discovered the SU3/3-fund structure unbiased) via the new
  `--init-from` (weights-only load; iters/optimizer/best reset, entropy+novelty
  schedules restart from 0 to re-inflate exploration from the plateau). Tests
  "given the structure, can gentle novelty cross the valley" at ~1/10th the
  compute. The warm-start is *not* a hint — f1nostop found that structure itself.

---

## Phase 8 — Warm start done properly (`f1ws`)

**What.** Same Phase-7 bundle (no-STOP, random start, max_steps 48, rollout 1024,
epochs 2) but: seed = f1nostop `ckpt_004500.pt` (best 50-iter mean return 0.71 at
entropy 0.79; `final.pt` came after an entropy collapse to 0.19), `ent_coef` held
**fixed at 0.01** (`--ent-anneal-iters 0`, ≈ the 0.0075–0.010 the seed was trained
under), **novelty off**. Isolates the one question Phase 7 could not answer:
does a gently perturbed plateau policy cross the ~0.08 charge valley on its own?

**Outcome (job 43908658, 2026-09-02, COMPLETED all 5000 iters in 6h). Failed, cleanly.**

| | value |
|---|---|
| best | 0.718 at iter 0 → 0.722 at iter 642 → flat for the remaining 4358 iters |
| exact hits | 0 |
| entropy | 0.4–0.7 throughout (held; never re-inflated, never collapsed except one blip at 4600) |
| mean episode return | 0.68–0.72, flat |
| new configs | 4,787 cache misses in ~5.1M env steps (99.9% hit rate → ~4 s/iter) |

Best model `SU3[r0, r1, r2]` (singlet + fundamental + sextet, all spin-½). Note
this is *not* the 3-fundamental 0.786 model: that was a single sample at iter
1138 of f1nostop; the policy f1nostop actually converged to (and that ckpt_4500
encodes) is this 0.72 singlet/fund/sextet family. The "structure found" claim in
Phase 4 was about the best sample, not the policy's mode.

What this establishes: with the entropy coefficient at the level the seed was
trained under, PPO sits on its plateau indefinitely, cycling through the same
few thousand configs and never sampling a multi-step charge/rep change. With the
coefficient high (Phases 6–7) it never concentrates at all. There is no entropy
setting in between that both concentrates *and* takes coordinated multi-step
excursions — because the excursion is a specific sequence of ±1 increments whose
first step is penalised by the per-step delta reward. Optimizer tuning is
exhausted; the fix has to be structural (action space or search).

**Informed next.** Two structural options were identified (see Phase 7 discussion):
(a) **set-charge actions** — replace the ±1 charge increments with "set charge c
of the cursor parton to value v", so the valley crossing becomes 1–2 actions
instead of a specific multi-step sequence; (b) **self-imitation** — replay the
best trajectories found so far and add an imitation loss so a rare 0.786+ sample
is consolidated instead of forgotten. Beyond those: MCTS on the deterministic
simulator (lookahead depth 2–3 sees past the valley) or Go-Explore.

---

## Phase 9 — Structural: set-charge actions + cursor (`f1set`)

**What.** Phases 6–8 exhausted the optimizer-side levers, so this changes the
*action space* and nothing else:
- `EnvConfig.charge_action_mode="set"` (`--charge-actions set`): the 8 ±1 charge
  increments become one action per lattice value per charge (13+13+13+25 = 64).
  Any charge value is one move away, so a multi-unit fix is no longer a path of
  scored, lower-scoring intermediates. **The lattice is unchanged** (`charge_max`
  6, steps 1/1/1/0.5). Tightening it toward SM values was considered and
  rejected as an SM prior: the only thing we assume known is the hadron content.
- `EnvConfig.use_cursor=True` (`--cursor`): `CURSOR_PREV/NEXT` so an edit can
  target any parton. **Discovered while sizing the head (2026-09-02):** every run
  so far edited the *last* parton only, so fixing parton 1 of 3 meant REMOVE ×2,
  edit, re-ADD and rebuild ×2. The Phase-5 diagnostic walked the charge lattice
  directly and measured a 2-step, 0.08-deep valley; in the actual action space
  that crossing was a long detour through 1- and 2-parton models. This, more than
  any entropy setting, is why f1ws never left its plateau.

Head 18 → 76. Mock check: exact SM reachable in 19 actions from an empty SU(3)
model (3 ADD, 3 SPIN, 9 charge sets, 4 cursor moves). Fresh run (head shape
changed, and unbiased); the f1nostop recipe that is known to concentrate
(epochs 4, entropy 0.03→0.005 over 5000, no novelty) with the Phase-7 horizon
(max_steps 48, rollout 1024). 15/15 unit tests incl. a new set/cursor test.

**Outcome, attempt 1 (job 44389130, 2026-09-04): OOM-killed at iter 38 of 5000**
(3.4 h, 24 GB, before the first checkpoint). Not a result about learning — at
iter 38 it was at best 0.462 with entropy 3.6 (max ≈4.2), i.e. still random, as
expected. What it exposed is an infrastructure failure the new action space
triggers:
- The persisted `SpectrumCache` is keyed on `canonical_key`, which **includes
  charges**. Under ±1 increments earlier runs stayed near the origin of the
  charge lattice and the shared file (2.47M entries, 5 GB) gave 99.9 % hits; under
  set-charge a random policy lands anywhere on the 13³×25 lattice, so ~800 of
  every 1024 steps were novel (hit rate 0.21).
- Every 200 novel entries the cache flushed with merge-on-flush, which reloads
  the *entire* 5 GB pickle beside the ~20 GB in-memory copy: a 2–3× RAM spike
  four times per iteration. All four recent jobs peaked at 22–23 GB on a 24 GB
  request; this one crossed. The reload/dump was also most of the 5 min/iter.
- The expensive call, `GetSpins`, depends only on (group, reps, spins, flavour
  pattern): it is **charge-free**, and its memo was in-process only, so every run
  re-paid it and the charge-decorated results were what got persisted.

**Fix (2026-09-08).** Canonicalize the `GetSpins` memo key (permutation +
flavour-relabel invariant) and persist *that* (`save_memo/load_memo`,
`--spins-memo` / `$SM_RL_SPINS_MEMO`, ~25 KB for 470 entries). Stop loading the
big spectra pickle; the in-memory spectrum cache is bounded (`--cache-max-entries`
200k, cleared when full). Validated against the live engine
(`scripts/validate_spins_memo.py`): GetSpins symmetric under 28 permuted /
relabelled queries; memoized spectra == raw spectra for 24 random models across
SU2/SU3/SU4/SO5/SP4/G2; save/load round-trip reproduces spectra with 0 new
calls. Login-node smoke: 128 env steps in ~5 s incl. 122 live GetSpins calls.
Expect iterations to be seconds, not minutes, once the memo is warm.

**Outcome, attempt 2 (job 45462268, 2026-09-08): completed 5000 iters in 4h44m
(cache fix held: 8,857 live GetSpins calls all run, memo 646 KB). Failed —
and the offline evaluator explains exactly how.**

| | value |
|---|---|
| best | 0.665 at iter 200 (G2, two fermions), flat for 4800 iters |
| exact | 0 |
| entropy | 2.7–3.0 for the whole run (max ≈4.2), even after `ent_coef` annealed to 0.005 |
| mean episode return | 0.06 → 0.50 by iter 1000, flat |

`scripts/eval_policy.py` (new: real-engine rollouts of a checkpoint, cheap now
the memo is persisted) on `final.pt`, 300 sampled + 16 greedy episodes:
- **Every** episode ends at SU(N) with N≥5 (greedy: SU(20)), three fermions with
  reps [singlet, singlet, fundamental], score **0.549 ± 0.00** regardless of
  start group. SU(3) is reached in 2/300 episodes.
- Entropy per step 2.8, but split: **0.14–0.29 over action *kinds*, 2.95 within
  `charge_set`**. The policy is decisive about *what* to do and uniform about
  *which value*.
- Direct probe: in SU(20) the lone fundamental's charges are **inert** (score
  0.549 for every EM value: its only ≤3-body singlets are its own q q̄, whose
  charges cancel). In SU(3) the same parton's charges matter (0.538–0.560).
- **71 % of the trained policy's steps leave the spectrum unchanged.** Policy
  entropy on those steps is **3.50**; on spectrum-changing steps it is **0.91**.
  86 % of the wasted steps are charge sets on the inert parton, 10 % are rank
  changes (SU(8)→SU(9) is also a no-op).

Mechanism — an **entropy-farming attractor**: with a per-step entropy bonus
`ec·H` and delta rewards that are 0 for a no-op, the highest-return trajectory
is "build something decent fast (0.549 by ~iter 200), make one parton inert by
ranking up, then spend the remaining ~35 steps uniformly re-setting its 64
charge actions" — collecting `ec·ln(64)` per step at zero cost. The set-charge
head handed the policy 64 mutually-equivalent actions per parton, which is far
more free entropy than the old ±1 pair; the entropy bonus paid it to park
there. Same family as the f1nostop/f1ws `[r0,r1,r2]` plateau (an inert singlet)
but much more lucrative. This is why entropy never fell: not exploration, but
exploitation of the regularizer. Also why the group drifts to big N (rank-up is
the cheapest way to make a parton inert), which is 17 rank steps from SU(3).

Bottom line for Phase 9: set-charge + cursor made the charge lattice reachable
(the target is 19 actions away) but, combined with entropy regularisation over
a flat 76-way head, created a degenerate optimum that dominates the SM basin.

**Informed next.** Remove the free-entropy profit rather than the action space:
- **No-op penalty** (recommended first): a small negative reward for any action
  that leaves the *spectrum* unchanged (compare spectra, not `canonical_key`,
  since inert-parton charge sets change the key but not the physics). Generic
  ("an action must do something"), no SM content, and it makes parking cost
  more than `ec·H` earns. Cursor moves are neutral by construction and would be
  taxed once each — cheap relative to a real edit's gain.
- **Factorised head / kind-level entropy**: regularise entropy over action
  *kinds* (or a hierarchical kind→value head) so 64 equivalent values are not 64×
  the entropy reward. More invasive (transformer head + PPO `evaluate`).
- Not recommended: `ent_coef=0` (f1fix showed early commitment) or capping the
  rank (treats the symptom; the start pool already encodes the rank≤4 physics).


**What to watch.** Max entropy is now ln(~70) ≈ 4.2, not 2.8, so expect a slower
fall; f1nostop dropped below 2 by iter ~1000. `best_score` past 0.786 and any
exact hits. If it concentrates on SU(3)/3-fermion structure and *still* sits
below 1.0, the next lever is self-imitation (consolidate rare good samples) or
MCTS on the simulator, not more action-space work.

---

### Config lineage (knobs introduced, in order)

`RewardConfig.metric=f1` + `normalization=physical` (P1) → `PPOConfig.start_mode`
`random`/`curriculum` + `EnvConfig.terminal_mode=graded` (P3) →
`EnvConfig.allow_stop=False` (P4) → `EnvConfig.novelty_coef` + CLI
`--max-steps/--epochs/--ent-coef-final` (P6) → `EnvConfig.novelty_mode` +
`PPOConfig.novelty_anneal_iters` (P7) → `train.py --init-from` warm start (P8) →
`EnvConfig.charge_action_mode=set` + `use_cursor` via `--charge-actions`/`--cursor` (P9). All default to the *previous* behaviour,
so every earlier run reproduces.
