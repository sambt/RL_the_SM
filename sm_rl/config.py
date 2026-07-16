"""Configuration dataclasses for the SM-rediscovery RL framework.

All knobs live here so experiments are reproducible and the physics / env /
reward / algorithm layers stay decoupled.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
_MACOS_KERNEL = "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"


def default_kernel_path() -> str:
    """Locate the Wolfram kernel across platforms.

    Priority: $WOLFRAM_KERNEL -> a kernel binary on PATH (Linux clusters usually
    expose `WolframKernel` / `wolfram` / `math`) -> the macOS app-bundle path.
    Override explicitly with EngineConfig(kernel_path=...) or $WOLFRAM_KERNEL.
    """
    env = os.environ.get("WOLFRAM_KERNEL")
    if env:
        return env
    for name in ("WolframKernel", "wolfram", "MathKernel", "math"):
        found = shutil.which(name)
        if found:
            return found
    return _MACOS_KERNEL


def default_mathematica_dir() -> str:
    """GroupMath/spinSinglets directory: $WOLFRAM_MATH_DIR or the repo's Mathematica/."""
    return os.environ.get("WOLFRAM_MATH_DIR", os.path.join(_REPO_DIR, "Mathematica"))


@dataclass
class EngineConfig:
    """How to talk to the Mathematica/GroupMath physics kernel.

    On a cluster, either put the kernel on PATH, set $WOLFRAM_KERNEL /
    $WOLFRAM_MATH_DIR, or pass kernel_path / mathematica_dir explicitly.
    """

    kernel_path: str = field(default_factory=default_kernel_path)
    mathematica_dir: str = field(default_factory=default_mathematica_dir)
    # Files Get[]-loaded at session start, in order, relative to mathematica_dir.
    groupmath_file: str = "GroupMath.m"
    spinsinglets_file: str = "spinSinglets.m"
    verbose: bool = True


@dataclass
class GroupConfig:
    """Which gauge groups / reps are in scope and how the group ladder steps."""

    # Rank ceilings per classical family (rank, not N). SU(N): rank N-1.
    su_max_rank: int = 19        # SU(2..20)
    so_max_rank: int = 10        # SO(3..~20)
    sp_max_rank: int = 10        # Sp(2..~20), rank = n for Sp(2n)
    include_exceptionals: bool = True   # G2, F4, E6, E7, E8
    # Largest rep dimension kept per group (reps beyond this are clamped away).
    # 248 = dim of the smallest non-trivial E8 rep, a natural default ceiling.
    max_rep_dim: int = 248


@dataclass
class EnvConfig:
    """MDP knobs for the model-building environment."""

    n_u1: int = 4                # number of U(1) charges per parton [EM, B, S, I3]
    max_partons: int = 8         # hard cap on matter fields (target needs 3)
    d_max: int = 3               # max fields per bound-state operator (super-linear cost)
    max_steps: int = 128         # episode step cap (truncation)

    # Charge action lattice: MODIFY steps charge c by charge_steps[c];
    # |charge_c| <= charge_max. Per-column so EM/B/S move on integers (scaled
    # convention) while I3 moves on half-integers.
    charge_steps: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5)
    charge_max: float = 6.0

    # CPT: each matter field implies its conjugate (dual rep, negated charges).
    # This is how "antiparticles implied by CPT" enters the spectrum computation,
    # so the (u,d,s) content alone can produce mesons and antibaryons.
    cpt_auto_conjugate: bool = True

    # Edit target. Default: edits act on the most-recently-added parton (stack
    # discipline). If True, a movable cursor + CURSOR_PREV/NEXT actions are added.
    use_cursor: bool = False

    # Per-step reward shaping mode:
    #   "absolute"  -> r_t = score(s_t)                       (dense, may reward stalling)
    #   "delta"     -> r_t = score(s_t) - score(s_{t-1})      (potential-based, telescopes)
    reward_mode: str = "delta"
    terminal_bonus: float = 10.0  # added at STOP when the model exactly matches target

    # Mask STOP until the model has at least this many partons. 0 = STOP anytime.
    # Kept at 0 deliberately: the "premature STOP" collapse this was added to
    # suppress was not a policy pathology but a correct response to a reward that
    # decreased monotonically in the number of predicted hadrons (see f1.py).
    # With RewardConfig.metric="f1" adding partons is rewarded, so whether the
    # agent voluntarily builds up is now a live diagnostic of the reward -- don't
    # mask it away without re-checking that.
    min_partons_before_stop: int = 0


@dataclass
class RewardConfig:
    """Reward-metric selection and weights (metric is pluggable)."""

    # "count"     -> CountReward (missing/extra hadrons, matched by quantum numbers)
    # "hungarian" -> HungarianReward (optimal partial matching w/ graded distances)
    # "f1"        -> F1Reward (quality-weighted F1; bounded [0,1], 1 == exact)
    metric: str = "count"

    # CountReward weights.
    w_missing: float = 1.0
    w_extra: float = 1.0

    # HungarianReward / F1Reward weights.
    w_charge: float = 1.0
    w_spin: float = 2.0
    lam_miss: float = 1.5
    lam_spur: float = 1.5

    # F1Reward: match-acceptance radius. A (target, pred) pair earns gain
    # 1 - cost/match_radius, so pairs at or beyond this cost earn nothing.
    # NB HungarianReward uses lam_miss + lam_spur for the same role; keep them
    # consistent if comparing the two.
    match_radius: float = 3.0

    # Charge normalization applied to a spectrum before matching:
    #   "physical" -> divide each U(1) column by charge_scale, undoing the
    #                 notebook's scaled-integer convention so the charge quantum
    #                 is 1. Deterministic (unlike "gcd") and keeps near-misses
    #                 inside the match radius. Recommended.
    #   "gcd"   -> per-column divide by gcd of that spectrum's nonzero charges
    #              (matches the GT-Invariants notebook; data-dependent -- see docstring)
    #   "none"  -> compare raw summed charges. NB the target's charge quantum is
    #              then 3 (EM/B/S are physical x3), which exceeds a match_radius
    #              of 3.0 -- i.e. near-misses earn nothing. Avoid.
    normalization: str = "gcd"

    # Per-column divisors for normalization="physical", matching the
    # [EM, B, S, I3] convention: EM/B/S are physical x3; I3 is already physical.
    charge_scale: tuple[float, ...] = (3.0, 3.0, 3.0, 1.0)
    round_decimals: int = 6      # signatures rounded to this precision before hashing


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 0
