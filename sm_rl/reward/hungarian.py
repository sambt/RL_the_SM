"""Hungarian optimal-partial-matching reward (opt-in alternative to CountReward).

Builds an augmented square cost matrix and solves a one-to-one assignment so
that: matched pairs pay a graded charge+spin distance, unmatched target states
pay `lam_miss`, and unmatched predicted states pay `lam_spur`. Graded distances
give partial credit for near-misses (unlike the exact-signature count metric),
while the one-to-one constraint keeps overproduction penalized.

score = 1 - total_cost / (lam_miss * |target|)  in (-inf, 1], 1 == perfect.
"""
from __future__ import annotations

import numpy as np

from .base import MatchResult, RewardMetric
from ..physics.spectrum import normalize_spectrum

_BIG = 1e6


class HungarianReward(RewardMetric):
    def match(self, pred_spectrum: np.ndarray, target_spectrum: np.ndarray) -> MatchResult:
        from scipy.optimize import linear_sum_assignment

        p = normalize_spectrum(pred_spectrum, self.n_u1, self.cfg.normalization)
        t = normalize_spectrum(target_spectrum, self.n_u1, self.cfg.normalization)
        nt, npd = len(t), len(p)
        lam_miss, lam_spur = self.cfg.lam_miss, self.cfg.lam_spur
        c0 = lam_miss + lam_spur              # match-acceptance radius

        if nt == 0:
            # no target: cost is purely spurious predictions
            total = lam_spur * npd
            score = 1.0 - total / max(lam_miss, 1e-9)
            return MatchResult(float(score), npd == 0, {"matched": 0, "missing": 0, "extra": npd})

        # pairwise ground cost: w_charge * L1(charges) + w_spin * |spin|
        if npd > 0:
            dq = np.abs(t[:, None, : self.n_u1] - p[None, :, : self.n_u1]).sum(axis=2)
            ds = np.abs(t[:, None, self.n_u1] - p[None, :, self.n_u1])
            C = self.cfg.w_charge * dq + self.cfg.w_spin * ds
            C = np.where(C < c0, C, _BIG)     # refuse implausible matches
        else:
            C = np.zeros((nt, 0))

        size = nt + npd
        M = np.zeros((size, size))
        M[:nt, :npd] = C
        # target i unmatched -> its own miss-dummy column (npd + i)
        miss_block = np.full((nt, nt), _BIG)
        np.fill_diagonal(miss_block, lam_miss)
        M[:nt, npd:] = miss_block
        # pred j unmatched -> its own spur-dummy row (nt + j)
        spur_block = np.full((npd, npd), _BIG)
        np.fill_diagonal(spur_block, lam_spur)
        M[nt:, :npd] = spur_block
        # dummy-dummy block stays 0

        r, c = linear_sum_assignment(M)
        total = float(M[r, c].sum())
        # count real matches (top-left block, below the big threshold)
        matched = int(sum(1 for i, j in zip(r, c) if i < nt and j < npd and M[i, j] < _BIG))
        missing = nt - matched
        extra = npd - matched
        score = 1.0 - total / (lam_miss * nt)
        exact = (missing == 0 and extra == 0 and total < 1e-6)
        return MatchResult(
            float(score), bool(exact),
            {"matched": matched, "missing": int(missing), "extra": int(extra),
             "total_cost": total, "n_pred": npd, "n_target": nt},
        )
