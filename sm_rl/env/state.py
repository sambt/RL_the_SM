"""The environment state: a gauge group + an ordered list of matter partons.

Design decisions (from the addendum):
  * Partons live in an ORDERED list. Ordering is not physical; it exists so the
    policy (with positional encoding) can track "the most recently added"
    parton, which is the default edit target.
  * Identical-looking partons are NOT merged. Each parton carries a unique
    integer `flavour` tag -- the "dummy quantum number" that distinguishes two
    otherwise-identical fields (u vs u'). Flavours are therefore always distinct.
  * A parton stores its gauge rep as an INDEX into its group's dimension-ordered
    rep list (0 = trivial, 1 = fundamental, ...), so a group change just clamps
    the index.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class Parton:
    rep_index: int           # index into GroupLadder.reps(group)
    spin: int                # GetSpins convention: 1 = scalar (spin 0), 2 = fermion (spin 1/2)
    charges: list[float]     # U(1) charges, length = EnvConfig.n_u1  (e.g. [EM, B, S, I3])
    flavour: int             # unique id ("dummy quantum number"); never shared

    def copy(self) -> "Parton":
        return Parton(self.rep_index, self.spin, list(self.charges), self.flavour)


@dataclass
class Model:
    group: str                              # GroupMath symbol, e.g. "SU3"
    partons: list[Parton] = field(default_factory=list)
    cursor: int = -1                        # edit target index; -1 == last parton
    _flav_counter: int = 0                  # monotonic source of unique flavour ids

    def copy(self) -> "Model":
        m = Model(self.group, [p.copy() for p in self.partons], self.cursor, self._flav_counter)
        return m

    # ---- convenience ------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.partons)

    def new_flavour(self) -> int:
        self._flav_counter += 1
        return self._flav_counter

    def edit_index(self, use_cursor: bool) -> int:
        """Which parton an edit acts on: the cursor (if enabled) else the last."""
        if self.n == 0:
            return -1
        if use_cursor:
            return max(0, min(self.cursor, self.n - 1))
        return self.n - 1

    def edit_target(self, use_cursor: bool) -> Parton | None:
        i = self.edit_index(use_cursor)
        return self.partons[i] if i >= 0 else None

    def to_dict(self) -> dict:
        """JSON-serializable snapshot (for logging the best model found)."""
        return {
            "group": self.group,
            "partons": [
                {"rep_index": p.rep_index, "spin": p.spin, "charges": list(p.charges)}
                for p in self.partons
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        m = cls(group=d["group"])
        for p in d["partons"]:
            m.partons.append(Parton(p["rep_index"], p["spin"], list(p["charges"]), m.new_flavour()))
        return m

    def describe(self) -> str:
        ps = [f"r{p.rep_index}/s{p.spin}/q{[round(c, 2) for c in p.charges]}" for p in self.partons]
        return f"{self.group}[" + ", ".join(ps) + "]"
