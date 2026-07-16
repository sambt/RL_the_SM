"""Count-based reward (default): score = -(w_missing * #missing + w_extra * #extra).

A predicted state "matches" a target state iff their quantum-number signatures
(normalized charges + spin label) are equal. Multiplicity-aware: three target
states at one signature need three predicted states there. Missing and extra are
counted with multiplicity, so overproduction is penalized and one predicted
state cannot cover several targets. score == 0 iff the multisets are equal.
"""
from __future__ import annotations

import numpy as np

from .base import MatchResult, RewardMetric, signature_counts
from ..physics.spectrum import normalize_spectrum


class CountReward(RewardMetric):
    def match(self, pred_spectrum: np.ndarray, target_spectrum: np.ndarray) -> MatchResult:
        rd = self.cfg.round_decimals
        scale = getattr(self.cfg, "charge_scale", None)
        p = normalize_spectrum(pred_spectrum, self.n_u1, self.cfg.normalization, scale)
        t = normalize_spectrum(target_spectrum, self.n_u1, self.cfg.normalization, scale)
        pc = signature_counts(p, rd)
        tc = signature_counts(t, rd)

        missing = sum(max(0, tc[s] - pc.get(s, 0)) for s in tc)
        extra = sum(max(0, pc[s] - tc.get(s, 0)) for s in pc)
        score = -(self.cfg.w_missing * missing + self.cfg.w_extra * extra)
        exact = (missing == 0 and extra == 0)
        return MatchResult(
            score=float(score),
            exact=bool(exact),
            info={"missing": int(missing), "extra": int(extra),
                  "n_pred": int(len(p)), "n_target": int(len(t))},
        )
