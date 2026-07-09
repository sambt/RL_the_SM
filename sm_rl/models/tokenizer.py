"""Featurize an environment observation into fixed-width tensors for the policy.

Design (from the plan + addendum):
  * Reps are described to the network by GROUP-INVARIANT physical scalars
    (log dim, Dynkin index, Casimir, congruency/N-ality), never raw Dynkin
    labels -- so a rep's token means the same thing across groups and the
    encoding survives group changes. These are looked up (and cached) from the
    engine.
  * The gauge group gets its own token (family one-hot + rank + rep-count).
  * Partons keep their list order; the network adds positional encodings
    (ordering is bookkeeping, not physics) and an is-edit-target flag.

Everything here is CPU/numpy; it produces plain arrays the torch policy stacks.
"""
from __future__ import annotations

import numpy as np

from ..physics.groups import GroupLadder, FAMILY_ORDER

# feature widths
N_FAMILY = len(FAMILY_ORDER)          # 4
GROUP_FEAT = N_FAMILY + 2             # family one-hot + lie_rank + log(n_reps)
REP_FEAT = 5                          # log dim, dynkin, casimir, cc0, cc1


class Tokenizer:
    def __init__(self, engine, ladder: GroupLadder, n_u1: int):
        self.engine = engine
        self.ladder = ladder
        self.n_u1 = n_u1
        self.parton_feat = REP_FEAT + 2 + n_u1 + 1   # rep + spin one-hot(2) + charges + edit flag
        self._rep_cache: dict[tuple[str, int], np.ndarray] = {}

    # ---- group + rep features (cached) -----------------------------------

    def group_features(self, group: str) -> np.ndarray:
        spec = self.ladder.spec(group)
        fam = np.zeros(N_FAMILY, dtype=np.float32)
        fam[FAMILY_ORDER.index(spec.family)] = 1.0
        return np.concatenate([
            fam,
            np.array([spec.lie_rank / 8.0, np.log1p(self.ladder.n_reps(spec))], dtype=np.float32),
        ])

    def rep_features(self, group: str, rep_index: int) -> np.ndarray:
        key = (group, rep_index)
        if key not in self._rep_cache:
            spec = self.ladder.spec(group)
            rep = self.ladder.rep_at(spec, rep_index)
            dim = self.engine.dim(group, rep)
            dynkin = float(self.engine.dynkin_index(group, rep))
            casimir = float(self.engine.casimir(group, rep))
            cc = self.engine.conjugacy_class(group, rep)
            cc0 = float(cc[0]) if len(cc) > 0 else 0.0
            cc1 = float(cc[1]) if len(cc) > 1 else 0.0
            self._rep_cache[key] = np.array(
                [np.log1p(dim), dynkin, casimir, cc0, cc1], dtype=np.float32)
        return self._rep_cache[key]

    def parton_features(self, model, use_cursor: bool) -> np.ndarray:
        if model.n == 0:
            return np.zeros((0, self.parton_feat), dtype=np.float32)
        edit_i = model.edit_index(use_cursor)
        rows = []
        for i, p in enumerate(model.partons):
            rf = self.rep_features(model.group, p.rep_index)
            spin = np.array([1.0, 0.0] if p.spin == 1 else [0.0, 1.0], dtype=np.float32)
            q = np.asarray(p.charges, dtype=np.float32)
            edit = np.array([1.0 if i == edit_i else 0.0], dtype=np.float32)
            rows.append(np.concatenate([rf, spin, q, edit]))
        return np.stack(rows).astype(np.float32)

    # ---- batch API for the policy ----------------------------------------

    def featurize(self, obs, use_cursor: bool) -> dict:
        model = obs["model"]
        return {
            "group_feat": self.group_features(model.group),
            "parton_feat": self.parton_features(model, use_cursor),
            "mask": np.asarray(obs["mask"], dtype=bool),
            "n": model.n,
        }

    def collate(self, feats: list[dict]) -> dict:
        """Pad a list of featurize() dicts into batched numpy arrays."""
        B = len(feats)
        nmax = max(1, max(f["n"] for f in feats))
        pf = np.zeros((B, nmax, self.parton_feat), dtype=np.float32)
        pad = np.ones((B, nmax), dtype=bool)          # True == padding (ignored)
        for b, f in enumerate(feats):
            n = f["n"]
            if n > 0:
                pf[b, :n] = f["parton_feat"]
                pad[b, :n] = False
        return {
            "group_feat": np.stack([f["group_feat"] for f in feats]).astype(np.float32),
            "parton_feat": pf,
            "pad_mask": pad,
            "action_mask": np.stack([f["mask"] for f in feats]),
            "n": np.array([f["n"] for f in feats]),
        }
