"""A tiny hardcoded Engine for Mathematica-free unit tests.

Covers SU2 and SU3 with dimension-ordered reps. `get_spins` is a deterministic
placeholder (real singlet physics is validated separately against the Wolfram
engine in scripts/phase0_validate.py), so mock-based tests exercise plumbing,
not representation theory.
"""
from __future__ import annotations

from sm_rl.physics.engine import Engine

_REPS = {
    # dimension-ordered: (rep dynkin) -> dim
    "SU2": [((0,), 1), ((1,), 2), ((2,), 3), ((3,), 4)],
    "SU3": [((0, 0), 1), ((1, 0), 3), ((0, 1), 3), ((2, 0), 6), ((0, 2), 6), ((1, 1), 8)],
}
_CONJ = {
    "SU2": {(0,): (0,), (1,): (1,), (2,): (2,), (3,): (3,)},
    "SU3": {(0, 0): (0, 0), (1, 0): (0, 1), (0, 1): (1, 0),
            (2, 0): (0, 2), (0, 2): (2, 0), (1, 1): (1, 1)},
}


class MockEngine(Engine):
    def start(self): return self
    def stop(self): pass

    def reps_up_to_dim(self, group, max_dim):
        return tuple(r for r, d in _REPS[group] if d <= max_dim)

    def dim(self, group, rep):
        return dict(_REPS[group])[tuple(rep)]

    def dynkin_index(self, group, rep):
        return float(self.dim(group, rep)) / 6.0

    def casimir(self, group, rep):
        return float(self.dim(group, rep)) / 3.0

    def conjugate_rep(self, group, rep):
        return _CONJ[group][tuple(rep)]

    def conjugacy_class(self, group, rep):
        # SU(N) N-ality: sum of Dynkin labels weighted by position, mod N.
        N = int(group[2:])
        k = sum((i + 1) * a for i, a in enumerate(rep)) % N
        return (k,)

    def adjoint(self, group):
        return {"SU2": (2,), "SU3": (1, 1)}[group]

    def get_spins(self, group, reps, spins, flavours):
        # deterministic placeholder: a lone trivial rep is a spin-carrying singlet.
        if len(reps) == 1 and all(x == 0 for x in reps[0]):
            return ((spins[0], 1),)
        return ()

    def min_copies_for_singlet(self, group, rep, cutoff=50):
        return {"SU2": 2, "SU3": 3}[group]
