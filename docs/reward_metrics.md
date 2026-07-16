# Reward metrics: the Hungarian score and the F1 correction

How `sm_rl` scores a predicted hadron spectrum against the Standard Model target.
Both metrics live in `sm_rl/reward/`; this document explains them explicitly — in
terms of the hadrons being matched, and mathematically — and shows why the
Hungarian score fails as a learning signal and how the F1 score fixes it.

All numeric examples below are computed against the **real 54-hadron target**
(SU(3) with `(u, d, s)`), so the tables are reproducible, not illustrative.

---

## 1 · The objects being scored

Both the prediction and the SM target are **sets of hadrons**. Each hadron is a
point in a 5-dimensional space — four U(1) charges plus a spin label:

$$
h = \bigl[\ \underbrace{EM,\ B,\ S,\ I_3}_{\text{4 charges}},\ \underbrace{2S+1}_{\text{spin label}}\ \bigr]
$$

- The SM target `T` has `n_t = 54` hadrons (the light-hadron spectrum).
- A model predicts a set `P` of `n_p` hadrons (its gauge-singlet bound states).

Charges use the notebook's scaled-integer convention: `EM, B, S` are stored as
**physical value × 3**, and `I_3` in halves. The spin label is `2S+1` (spin 0 → 1,
½ → 2, 1 → 3, 3/2 → 4).

Scoring asks: **how well can predicted hadrons be paired one-to-one with SM
hadrons?**

---

## 2 · The Hungarian matching score (`reward/hungarian.py`)

### 2.1 Cost of pairing two hadrons

For a target hadron `t_i` and a predicted hadron `p_j`, the ground cost is an L1
distance in charge space plus a weighted spin mismatch:

$$
c_{ij} \;=\; w_\text{charge}\sum_{k=1}^{4}\bigl\lvert q^{(t_i)}_k - q^{(p_j)}_k\bigr\rvert
\;+\; w_\text{spin}\,\bigl\lvert (2S{+}1)_{t_i} - (2S{+}1)_{p_j}\bigr\rvert
$$

with defaults `w_charge = 1`, `w_spin = 2`. Two hadrons with identical quantum
numbers cost 0; the further apart, the more it costs to call them "the same
particle."

**Acceptance radius.** A hard cutoff

$$
c_0 \;=\; \lambda_\text{miss} + \lambda_\text{spur} \;=\; 1.5 + 1.5 \;=\; 3.0
$$

*refuses* any pair with `c_ij ≥ c0` (its cost is set to ∞). Two hadrons more than
`c0` apart are too different to be "the same particle," so the matcher will never
pair them.

### 2.2 The assignment problem

Not every hadron gets a partner. An SM hadron the model fails to produce is
**missing**; a predicted hadron with no SM counterpart is **spurious / extra**.
This is encoded as one square cost matrix `M` of size `(n_t + n_p) × (n_t + n_p)`:

$$
M=\begin{bmatrix}
C & \textbf{miss-dummies}\\[4pt]
\textbf{spur-dummies} & \mathbf{0}
\end{bmatrix}
$$

- **top-left `C`** (`n_t × n_p`): the real pair costs `c_ij` (refused pairs = ∞).
- **top-right** (`n_t × n_t`): diagonal = `λ_miss = 1.5`, off-diagonal = ∞.
  Target `i` may pair with *its own* dummy column — i.e. be left **missing** — at
  price `λ_miss`.
- **bottom-left** (`n_p × n_p`): diagonal = `λ_spur = 1.5`, off-diagonal = ∞.
  Prediction `j` may be flagged **spurious** at price `λ_spur`.
- **bottom-right**: zeros (dummy-to-dummy pairing is free).

`scipy.optimize.linear_sum_assignment` finds the **minimum-cost one-to-one
matching** on `M`. Because every off-diagonal dummy entry is ∞, the optimum only
ever does one of three things with each hadron: match a real pair (if `c_ij < c0`),
leave an SM hadron missing (`λ_miss`), or flag a prediction spurious (`λ_spur`).

### 2.3 Total cost and score

Let `m` be the number of real pairs actually matched. Then
`missing = n_t − m`, `extra = n_p − m`, and the optimal cost decomposes cleanly:

$$
\boxed{\ \text{total} \;=\;
\underbrace{\sum_{\text{matched }(i,j)} c_{ij}}_{\text{graded near-miss cost}}
\;+\; \lambda_\text{miss}\,(n_t - m)
\;+\; \lambda_\text{spur}\,(n_p - m)\ }
$$

$$
\boxed{\ \text{score} \;=\; 1 - \frac{\text{total}}{\lambda_\text{miss}\cdot n_t}
\;=\; 1 - \frac{\text{total}}{81}\ }
$$

(the denominator is `λ_miss · n_t = 1.5 · 54 = 81`.) Perfect reproduction ⇒
`total = 0` ⇒ `score = 1`. The score lies in `(−∞, 1]`.

### 2.4 Worked numbers (real 54-hadron target)

| prediction | matched | missing | extra | total | score |
|---|---:|---:|---:|---:|---:|
| target vs itself | 54 | 0 | 0 | 0.00 | **+1.0000** |
| **empty** (predict nothing) | 0 | 54 | 0 | 81.00 | **0.0000** |
| half the target (27 exact) | 27 | 27 | 0 | 40.50 | +0.5000 |
| all hadrons shifted `EM +1` **raw** unit | 54 | 0 | 0 | 54.00 | +0.3333 |
| all hadrons shifted `EM +3` (**1 physical** unit) | 28 | 26 | 26 | 106.00 | −0.3086 |
| target + 1 junk hadron | 54 | 0 | 1 | 1.50 | +0.9815 |
| target + 10 junk | 54 | 0 | 10 | 15.00 | +0.8148 |
| target + 50 junk | 54 | 0 | 50 | 75.00 | +0.0741 |
| target + 200 junk | 54 | 0 | 200 | 300.00 | **−2.7037** |

Each row reads straight off the formula. Empty: `total = 1.5·54 = 81`, score
`1 − 81/81 = 0` — **the empty model is the zero baseline.** Each spurious hadron
adds exactly `λ_spur = 1.5` to the total, subtracting `1.5/81 ≈ 0.0185` from the
score, **with no floor**: 200 junk hadrons drives the score to −2.70.

### 2.5 Two structural problems

**(a) The score decreases monotonically in the number of predicted hadrons.**
In the "+ k junk" rows, `matched` is stuck at 54 while `extra` climbs 1-for-1. In a
real search it is worse: the number of predictions that can land within radius
`c0 = 3` of a *distinct* SM hadron **saturates** (≈ 11 for typical models), so
every prediction beyond that is pure spurious cost. Then

$$
\text{score}\;\approx\; 1 - \frac{81 - (\text{matched credit}) + 1.5\cdot\text{extra}}{81}
$$

is dominated by the `+1.5·extra` term. Measured over cached SU(3) models,
`corr(n_pred, score) = −0.995`. **The reward-maximising model predicts almost
nothing**, and SU(3) — the only group forming *many* singlets at `d_max = 3` — is
penalised hardest, ranking **27/30** among groups. The gradient points *away* from
the Standard Model.

**(b) Normalization is load-bearing.** Compare the two "shifted" rows. A `+1` *raw*
charge error costs 1.0, comfortably inside `c0 = 3`, so all 54 still match
(score +0.333). But a `+3` shift — **one physical unit**, the actual charge quantum,
since charges are stored ×3 — costs exactly 3.0, hits the rejection radius, and 26
of the 54 hadrons flip to missing + spurious (score −0.309). Under
`normalization="none"` the physical charge quantum sits right at the rejection
radius, so genuine near-misses earn **no** partial credit — the graded metric
behaves like exact-match-only, carrying no gradient toward the right charges.

> **Do not use `hungarian` + `normalization="none"` as a learning signal.** Two
> 5k-iteration training runs confirmed the failure: the free-group run never beat
> the score its *randomly initialised* policy achieved at iteration 0.

---

## 3 · The F1 correction (`reward/f1.py`)

The F1 metric keeps the same idea — optimally pair predicted hadrons to SM hadrons
in the same 5-D space — but changes **what an unmatched hadron costs** and **how the
result is normalised**, fixing both problems above.

### 3.1 Gain instead of cost

Using the same ground cost `c_ij` and radius `match_radius` (default 3.0), define a
**gain** in `[0, 1]` for each candidate pair:

$$
\boxed{\ g_{ij} \;=\; \max\!\left(0,\ 1 - \frac{c_{ij}}{\text{match\_radius}}\right)\ }
$$

An exact pair earns `g = 1`; a pair at the radius earns `g = 0`; beyond the radius,
`g = 0` (never negative). This is the key change: an unmatched hadron contributes
**zero**, never an unbounded penalty.

### 3.2 The assignment

Pair predictions to targets to **maximise total gain** — equivalently, minimise
`−g` via the same `linear_sum_assignment`, on a square `max(n_t, n_p)` matrix padded
with zero-gain entries (so surplus hadrons on either side simply go unmatched at no
cost). Let the optimal assignment collect

$$
\text{soft} \;=\; \sum_{\text{matched }(i,j)} g_{ij}
$$

— the "soft" count of matched hadrons (a perfect pair counts 1, a near-miss counts a
fraction, a non-pair counts 0).

### 3.3 The score — a quality-weighted F1

$$
\boxed{\ \text{score} \;=\; \frac{2\,\cdot\,\text{soft}}{n_p + n_t}\ \in [0, 1]\ }
$$

This is exactly the **F1 form** `2·(soft-matched) / (predicted + target)`, i.e. the
harmonic-mean combination of a soft precision `soft/n_p` and soft recall
`soft/n_t`. Properties:

- **Bounded in `[0, 1]`**, and `score = 1` **iff** the prediction is exact
  (`n_p = n_t` and every pair has `g = 1`).
- **Empty prediction → 0** (soft = 0), same zero baseline as Hungarian.
- **Overproduction is penalised but bounded.** Adding junk grows `n_p` in the
  denominator: precision falls, but the score decays *toward 0*, never diverging.
- **Matching more of the target always helps** — nothing punishes predicting the
  hadron that opens the baryon sector.

### 3.4 Same worked examples, side by side

| prediction | Hungarian (`none`) | F1 (`physical`) |
|---|---:|---:|
| target vs itself | +1.0000 | **+1.0000** |
| empty | 0.0000 | **0.0000** |
| half the target (27 exact) | +0.5000 | **+0.6667** |
| all shifted `EM +3` (1 phys unit) | −0.3086 | **+0.6667** |
| target + 1 junk | +0.9815 | +0.9908 |
| target + 10 junk | +0.8148 | +0.9153 |
| target + 50 junk | +0.0741 | +0.6835 |
| target + 200 junk | **−2.7037** | **+0.3506** |

Read the contrast:

- **Overproduction stays sane.** 200 junk hadrons: Hungarian −2.70 (worse than
  empty), F1 +0.35 (still recognises all 54 SM hadrons are present). For F1,
  `soft = 54`, `n_p = 254`, `n_t = 54` ⇒ `2·54/(254+54) = 0.3506`.
- **Near-misses earn graded credit.** The `EM +3` row (every hadron one physical
  unit off): Hungarian collapses to −0.31; F1 gives +0.667, because
  `normalization="physical"` divides charges by `(3, 3, 3, 1)` so the quantum is 1
  and a one-unit miss lands at `g = 1 − 1/3 = 2/3`.

### 3.5 Why `physical` normalization matters here too

`normalization="physical"` divides each charge column by `charge_scale = (3, 3, 3,
1)`, undoing the ×3 convention so the charge quantum is **1** rather than 3. This is
deterministic (unlike `gcd`, which rescales per-spectrum and is data-dependent) and
puts one-unit near-misses safely inside the radius. A single isolated target hadron,
shifted by whole physical units of EM, shows the resulting gradient:

| prediction | `g` = score |
|---|---:|
| exact | +1.0000 |
| `EM +1` physical unit | +0.6667 |
| `EM +2` physical units | +0.3333 |
| `EM +3` physical units | 0.0000 |

A smooth ramp — exactly the graded signal PPO needs and the Hungarian/`none`
combination destroys.

### 3.6 Net effect on the search

Rescored on the cached spectra (no Mathematica time), the F1 + `physical` change
moves every diagnostic in the right direction:

| quantity | Hungarian + `none` | F1 + `physical` |
|---|---:|---:|
| `corr(n_pred, score)` over SU(3) models | −0.995 | +0.272 |
| SU(3) rank among 30 groups | 27 / 30 | 6 / 30 |
| fraction of SU(3) models beating the empty model | 17 % | 100 % |
| does adding a quark raise the score? | no (every quark hurts) | yes |

In training, the seeded SU(3) run under F1 reached **score 1.0 — an exact
rediscovery** of the `(u, d, s)` spectrum, and the free-group run climbed to 0.73 on
an SU(3) model, versus 0.14 on SU5/SU8 under the old reward.

---

## 4 · Reference

| symbol | meaning | default |
|---|---|---|
| `n_t`, `n_p` | number of target / predicted hadrons | `n_t = 54` |
| `w_charge`, `w_spin` | ground-cost weights (charge L1, spin) | 1.0, 2.0 |
| `λ_miss`, `λ_spur` | Hungarian miss / spurious penalties | 1.5, 1.5 |
| `c0` | Hungarian acceptance radius, `λ_miss + λ_spur` | 3.0 |
| `match_radius` | F1 gain radius | 3.0 |
| `charge_scale` | `physical` normalization divisors `[EM, B, S, I3]` | (3, 3, 3, 1) |

- Hungarian: `score = 1 − total / (λ_miss · n_t)`, range `(−∞, 1]`.
- F1: `score = 2 · soft / (n_p + n_t)`, range `[0, 1]`, with
  `g_ij = max(0, 1 − c_ij / match_radius)`.

Both are validated by regression tests in `tests/test_core.py`
(`test_f1_target_is_optimal`, `test_f1_predicting_more_beats_predicting_nothing`,
`test_f1_bounded_under_overproduction`,
`test_f1_partial_credit_needs_physical_normalization`).
