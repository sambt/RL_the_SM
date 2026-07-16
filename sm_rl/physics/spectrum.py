"""Turn a Model into its predicted gauge-singlet bound-state spectrum.

This is a faithful port of `get_spectrum` from the GT-Invariants notebook, with
two additions driven by the addendum:

  * CPT: each matter parton optionally contributes its conjugate (dual rep,
    negated charges) to the field list, so that e.g. the (u,d,s) content alone
    produces mesons (q qbar) and antibaryons -- "antiparticles implied by CPT".
  * Canonical caching: the expensive per-operator GetSpins calls are memoized on
    the engine, and whole spectra are memoized on a canonical model key so that
    revisited states (very common under add/remove/modify) cost nothing.

A spectrum is an (M, n_u1 + 1) float array: columns [charges..., su2_spin_label].
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement
from math import gcd

import numpy as np

from .engine import Engine, Rep
from .groups import GroupLadder

ResolvedField = tuple[Rep, int, tuple[float, ...]]   # (rep dynkin, spin, charges)


def resolve_fields(model, ladder: GroupLadder, cpt_auto_conjugate: bool) -> list[ResolvedField]:
    """Resolve parton rep-indices to Dynkin labels and (optionally) append the
    CPT conjugate of each parton."""
    spec = ladder.spec(model.group)
    fields: list[ResolvedField] = []
    for p in model.partons:
        rep = ladder.rep_at(spec, p.rep_index)
        q = tuple(float(x) for x in p.charges)
        fields.append((rep, p.spin, q))
        if cpt_auto_conjugate:
            conj = tuple(ladder.engine.conjugate_rep(spec.gm, rep))
            qc = tuple(-x for x in q)
            # A truly self-conjugate, neutral field is its own antiparticle;
            # do not double-count it.
            if not (conj == tuple(rep) and all(x == 0.0 for x in q)):
                fields.append((conj, p.spin, qc))
    return fields


def canonical_key(model, ladder: GroupLadder, cpt_auto_conjugate: bool, round_decimals: int = 6):
    """Order-invariant hashable key identifying the physics of a model."""
    fields = resolve_fields(model, ladder, cpt_auto_conjugate)
    core = sorted(
        (tuple(rep), int(spin), tuple(round(x, round_decimals) for x in q))
        for rep, spin, q in fields
    )
    return (model.group, tuple(core))


def compute_spectrum(engine: Engine, group: str, fields: list[ResolvedField],
                     d_max: int, n_u1: int) -> np.ndarray:
    """Enumerate all <= d_max-parton operators and collect gauge singlets."""
    n = len(fields)
    if n == 0:
        return np.zeros((0, n_u1 + 1))
    reps = [f[0] for f in fields]
    spins = [f[1] for f in fields]
    charges = [np.asarray(f[2], dtype=float) for f in fields]
    flavours = list(range(1, n + 1))          # every field distinct (see state.py)

    rows: list[np.ndarray] = []
    for d in range(1, d_max + 1):
        for combo in combinations_with_replacement(range(n), d):
            idx = list(combo)
            invariants = engine.get_spins(
                group, [reps[i] for i in idx], [spins[i] for i in idx], [flavours[i] for i in idx]
            )
            if invariants:
                op_charge = np.sum([charges[i] for i in idx], axis=0)
                for label, mult in invariants:
                    rows.extend(np.append(op_charge, label) for _ in range(mult))
    if not rows:
        return np.zeros((0, n_u1 + 1))
    return np.asarray(rows, dtype=float)


def spectrum_of(engine, ladder, model, env_cfg, cache: dict | None = None) -> np.ndarray:
    """Raw (un-normalized) predicted spectrum for `model`, memoized on the
    canonical key. Normalization is the reward layer's responsibility."""
    key = canonical_key(model, ladder, env_cfg.cpt_auto_conjugate)
    if cache is not None and key in cache:
        return cache[key]
    fields = resolve_fields(model, ladder, env_cfg.cpt_auto_conjugate)
    spec = compute_spectrum(engine, model.group, fields, env_cfg.d_max, env_cfg.n_u1)
    if cache is not None:
        cache[key] = spec
    return spec


# ---- charge normalization -------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else max(a, b)


def _rational_col_gcd(values: np.ndarray, max_den: int = 10**6) -> Fraction | None:
    """Largest g such that every nonzero value / g is an integer (rational gcd)."""
    fr = [Fraction(float(v)).limit_denominator(max_den) for v in values if abs(v) > 0]
    if not fr:
        return None
    den = 1
    for f in fr:
        den = _lcm(den, f.denominator)
    g = 0
    for f in fr:
        g = gcd(g, abs(int(f * den)))
    return Fraction(g, den) if g else None


def normalize_spectrum(spec: np.ndarray, n_u1: int, mode: str = "gcd",
                       scale: tuple[float, ...] | None = None) -> np.ndarray:
    """Per-column charge normalization applied before matching.

    mode="physical": divide column c by scale[c], undoing the notebook's
    scaled-integer convention (EM/B/S are physical x3) so the charge quantum
    becomes 1. Deterministic and prediction-independent, unlike "gcd". This
    matters for graded metrics: with mode="none" the target's charge quantum is
    3, so a one-unit charge error already costs more than a match radius of 3
    and earns zero partial credit.

    mode="gcd": divide each U(1) column by the rational gcd of its nonzero
    entries (so the smallest charge unit becomes 1) -- matches the GT-Invariants
    notebook. NB this is data-dependent: adding/removing a predicted state can
    rescale a whole column, so target and prediction must be normalized by the
    SAME rule (they are).

    mode="none": raw charges, no rescaling.
    """
    if spec.shape[0] == 0 or mode == "none":
        return spec
    out = spec.astype(float).copy()
    if mode == "physical":
        if scale is None:
            raise ValueError("normalization='physical' requires a charge scale")
        for c in range(min(n_u1, len(scale))):
            if scale[c]:
                out[:, c] = out[:, c] / float(scale[c])
        return out
    for c in range(n_u1):
        g = _rational_col_gcd(out[:, c])
        if g is not None and g != 0:
            out[:, c] = out[:, c] / float(g)
    return out
