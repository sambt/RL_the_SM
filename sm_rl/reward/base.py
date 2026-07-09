"""Reward-metric interface. A metric scores a predicted spectrum against the
target; higher is better. Metrics are pluggable (see RewardConfig.metric)."""
from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    score: float               # higher = better (CountReward: <= 0, 0 == perfect)
    exact: bool                # prediction reproduces the target exactly
    info: dict = field(default_factory=dict)


def signature_counts(spec: np.ndarray, round_decimals: int) -> Counter:
    """Counter over (rounded charges..., spin-label) rows of a spectrum."""
    return Counter(tuple(np.round(row, round_decimals)) for row in spec)


class RewardMetric(abc.ABC):
    """Scores pred vs target. Owns its own charge normalization."""

    def __init__(self, cfg, n_u1: int):
        self.cfg = cfg
        self.n_u1 = n_u1

    @abc.abstractmethod
    def match(self, pred_spectrum: np.ndarray, target_spectrum: np.ndarray) -> MatchResult: ...


def make_metric(cfg, n_u1: int) -> RewardMetric:
    from .count import CountReward
    from .hungarian import HungarianReward

    if cfg.metric == "count":
        return CountReward(cfg, n_u1)
    if cfg.metric == "hungarian":
        return HungarianReward(cfg, n_u1)
    raise ValueError(f"unknown reward metric: {cfg.metric!r}")
