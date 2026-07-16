"""Quality-weighted F1 reward -- the default for new runs.

Why this exists (see also HungarianReward, which this replaces as the default):
HungarianReward charges an *additive, unbounded* `lam_spur` for every predicted
state that fails to match, while the number of states it can match saturates.
Its score is therefore essentially a decreasing linear function of how many
hadrons a model predicts (measured: corr(n_pred, score) = -0.995 over cached
SU(3) models), so the reward-maximising model is one that predicts nothing --
and SU(3), the only group that forms many singlets at d_max=3, scores *worst*
of any group on the ladder. That gradient points away from the Standard Model.

This metric instead scores a quality-weighted F1:

    gain(i, j) = max(0, 1 - cost(i, j) / match_radius)     in [0, 1]
    soft       = sum of gain over an optimal one-to-one assignment
    score      = 2 * soft / (n_pred + n_target)            in [0, 1]

Overproduction dilutes precision sub-linearly instead of adding unbounded cost,
so the score is bounded, 1.0 iff exact, and 0.0 for both an empty prediction and
for unbounded spam. Matching *more* target states always helps, so adding the
quark that opens the baryon sector is rewarded rather than punished.

Pair with normalization="physical": the graded gain only produces a usable
gradient if near-misses fall inside match_radius, which requires the charge
quantum to be 1 rather than the raw convention's 3.
"""
from __future__ import annotations

import numpy as np

from .base import MatchResult, RewardMetric
from ..physics.spectrum import normalize_spectrum


class F1Reward(RewardMetric):
    def match(self, pred_spectrum: np.ndarray, target_spectrum: np.ndarray) -> MatchResult:
        from scipy.optimize import linear_sum_assignment

        scale = getattr(self.cfg, "charge_scale", None)
        p = normalize_spectrum(pred_spectrum, self.n_u1, self.cfg.normalization, scale)
        t = normalize_spectrum(target_spectrum, self.n_u1, self.cfg.normalization, scale)
        nt, npd = len(t), len(p)
        if nt + npd == 0:
            return MatchResult(0.0, True, {"matched": 0, "missing": 0, "extra": 0, "n_pred": 0})
        if nt == 0 or npd == 0:
            return MatchResult(0.0, False, {"matched": 0, "missing": nt, "extra": npd,
                                            "n_pred": npd, "n_target": nt})

        radius = float(self.cfg.match_radius)
        dq = np.abs(t[:, None, : self.n_u1] - p[None, :, : self.n_u1]).sum(axis=2)
        ds = np.abs(t[:, None, self.n_u1] - p[None, :, self.n_u1])
        cost = self.cfg.w_charge * dq + self.cfg.w_spin * ds
        gain = np.maximum(0.0, 1.0 - cost / radius)

        # square-pad so unassigned rows/cols simply earn zero gain
        size = max(nt, npd)
        M = np.zeros((size, size))
        M[:nt, :npd] = -gain                       # linear_sum_assignment minimises
        r, c = linear_sum_assignment(M)
        soft = float(-M[r, c].sum())
        matched = int(sum(1 for i, j in zip(r, c)
                          if i < nt and j < npd and gain[i, j] > 1e-9))

        score = 2.0 * soft / (nt + npd)
        exact = bool(nt == npd and abs(soft - nt) < 1e-6)
        return MatchResult(
            float(score), exact,
            {"matched": matched, "missing": int(nt - matched), "extra": int(npd - matched),
             "soft_matched": soft, "n_pred": npd, "n_target": nt},
        )
