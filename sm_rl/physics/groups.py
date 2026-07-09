"""The gauge-group ladder.

Groups are addressed as GroupMath strings ("SU3", "SO10", "SP6", "E6", ...).
The agent moves between them with two action families:

  * CHANGE_RANK : step to the neighbouring group of the *same* type, i.e. the
                  next/previous N in SU(N)/SO(N)/Sp(N) (or the rank-adjacent
                  exceptional).
  * CHANGE_TYPE : switch classical family (SU <-> SO <-> Sp <-> exceptional),
                  mapping to the group of *nearest Lie rank* in the new family.
                  Exceptionals only exist at ranks {2,4,6,7,8}, so this rounds.

Representations of each group are dimension-ordered (index 0 = trivial,
1 = fundamental, ...). A parton stores a rep *index*; whenever the group
changes the index is clamped down to the new group's largest admissible rep
(<= max_rep_dim), exactly as the addendum specifies (e.g. r3 of SU(3) -> r1 of
E8, since only the trivial and the 248 survive the dimension cap for E8).
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine, Rep
from ..config import GroupConfig

FAMILY_ORDER = ("SU", "SO", "SP", "EX")


@dataclass(frozen=True)
class GroupSpec:
    gm: str          # GroupMath symbol, e.g. "SU3"
    family: str      # "SU" | "SO" | "SP" | "EX"
    N: int           # the N in SU(N)/SO(N)/Sp(N); for EX, 0
    lie_rank: int    # rank of the Lie algebra

    def __str__(self) -> str:
        return self.gm


def _su_specs(max_rank: int) -> list[GroupSpec]:
    # SU(N), N = 2..max_rank+1, lie rank N-1.
    return [GroupSpec(f"SU{N}", "SU", N, N - 1) for N in range(2, max_rank + 2)]


def _so_specs(max_rank: int) -> list[GroupSpec]:
    # SO(N), N >= 3, skip SO(4) (semisimple). lie rank = floor(N/2).
    out = []
    N = 3
    while N // 2 <= max_rank:
        if N != 4:
            out.append(GroupSpec(f"SO{N}", "SO", N, N // 2))
        N += 1
    return out


def _sp_specs(max_rank: int) -> list[GroupSpec]:
    # Sp(2n), n = 1..max_rank, lie rank n.
    return [GroupSpec(f"SP{2 * n}", "SP", 2 * n, n) for n in range(1, max_rank + 1)]


def _ex_specs() -> list[GroupSpec]:
    return [
        GroupSpec("G2", "EX", 0, 2),
        GroupSpec("F4", "EX", 0, 4),
        GroupSpec("E6", "EX", 0, 6),
        GroupSpec("E7", "EX", 0, 7),
        GroupSpec("E8", "EX", 0, 8),
    ]


class GroupLadder:
    """Registry of candidate groups + the type/rank transition graph, backed by
    an :class:`Engine` for the (cached) dimension-ordered rep lists."""

    def __init__(self, engine: Engine, cfg: GroupConfig | None = None):
        self.engine = engine
        self.cfg = cfg or GroupConfig()
        fams: dict[str, list[GroupSpec]] = {
            "SU": _su_specs(self.cfg.su_max_rank),
            "SO": _so_specs(self.cfg.so_max_rank),
            "SP": _sp_specs(self.cfg.sp_max_rank),
        }
        if self.cfg.include_exceptionals:
            fams["EX"] = _ex_specs()
        # Keep only present, nonempty families in canonical order.
        self.family_order = [f for f in FAMILY_ORDER if fams.get(f)]
        self.families = {f: fams[f] for f in self.family_order}
        self.specs = [s for f in self.family_order for s in self.families[f]]
        self._by_gm = {s.gm: s for s in self.specs}
        self._reps_cache: dict[str, tuple[Rep, ...]] = {}

    # ---- lookups ----------------------------------------------------------

    def spec(self, gm: str) -> GroupSpec:
        return self._by_gm[gm]

    @property
    def default(self) -> GroupSpec:
        """Starting group for an episode: SU(2), the smallest simple group."""
        return self.families["SU"][0]

    def reps(self, spec: GroupSpec) -> tuple[Rep, ...]:
        """Dimension-ordered irreps of `spec` with dim <= max_rep_dim."""
        if spec.gm not in self._reps_cache:
            self._reps_cache[spec.gm] = self.engine.reps_up_to_dim(spec.gm, self.cfg.max_rep_dim)
        return self._reps_cache[spec.gm]

    def n_reps(self, spec: GroupSpec) -> int:
        return len(self.reps(spec))

    def rep_at(self, spec: GroupSpec, index: int) -> Rep:
        return self.reps(spec)[self.clamp_index(spec, index)]

    def clamp_index(self, spec: GroupSpec, index: int) -> int:
        """Clamp a rep index into [0, n_reps-1]. On a group change this is what
        'rounds reps down' when the new group has fewer admissible reps."""
        return max(0, min(index, self.n_reps(spec) - 1))

    # ---- transitions ------------------------------------------------------

    def rank_step(self, spec: GroupSpec, direction: int) -> GroupSpec | None:
        """Neighbouring group of the same family (next/prev N, or rank-adjacent
        exceptional). Returns None at the family's boundary (mask this action)."""
        fam = self.families[spec.family]
        i = fam.index(spec) + direction
        if 0 <= i < len(fam):
            return fam[i]
        return None

    def type_step(self, spec: GroupSpec, direction: int) -> GroupSpec:
        """Switch to the next/previous family, landing on the group of nearest
        Lie rank there. Cycles through `family_order`."""
        fo = self.family_order
        fi = (fo.index(spec.family) + direction) % len(fo)
        target_family = fo[fi]
        candidates = self.families[target_family]
        # nearest lie_rank; ties -> smaller rank (smaller N / simpler group).
        return min(candidates, key=lambda s: (abs(s.lie_rank - spec.lie_rank), s.lie_rank, s.N))
