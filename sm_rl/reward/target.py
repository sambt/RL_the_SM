"""Build the rediscovery target: the QCD light-quark model and its hadron
spectrum, computed by the physics engine itself (so the target is guaranteed
self-consistent with predictions and the true model is the argmax).

Charges use the GT-Invariants notebook's scaled-integer convention
([EM, B, S, I3] with EM,B,S integers = physical value x 3, I3 in halves).
Antiquarks are supplied by CPT (EnvConfig.cpt_auto_conjugate), so the content
to rediscover is just the quarks (u, d, s).
"""
from __future__ import annotations

import numpy as np

from ..env.state import Model, Parton
from ..physics.groups import GroupLadder
from ..physics.spectrum import spectrum_of

# [EM, B, S, I3] in the scaled-integer convention. Antiquarks come from CPT.
QUARK_CHARGES: dict[str, list[float]] = {
    "u": [2.0, 1.0, 0.0, 0.5],
    "d": [-1.0, 1.0, 0.0, -0.5],
    "s": [-1.0, 1.0, -1.0, 0.0],
    "c": [2.0, 1.0, 0.0, 0.0],    # (extension; charm/bottom/top if ever wanted)
    "b": [-1.0, 1.0, 0.0, 0.0],
    "t": [2.0, 1.0, 0.0, 0.0],
}

FERMION = 2   # GetSpins spin label for spin-1/2


def fundamental_index(ladder: GroupLadder, group: str = "SU3") -> int:
    """Dimension-ordered index of the fundamental (smallest nontrivial) rep."""
    spec = ladder.spec(group)
    reps = ladder.reps(spec)
    # index 0 is trivial; the fundamental is the next entry.
    return 1 if len(reps) > 1 else 0


def build_target(engine, ladder: GroupLadder, env_cfg,
                 quarks: tuple[str, ...] = ("u", "d", "s"),
                 group: str = "SU3", cache: dict | None = None):
    """Return (target_spectrum, target_model) for the given quark content."""
    spec = ladder.spec(group)
    fund = fundamental_index(ladder, group)
    model = Model(group=group)
    for q in quarks:
        model.partons.append(
            Parton(rep_index=fund, spin=FERMION,
                   charges=list(QUARK_CHARGES[q]), flavour=model.new_flavour())
        )
    target = spectrum_of(engine, ladder, model, env_cfg, cache=cache)
    return target, model
