"""The model-building environment.

State: a gauge group + ordered parton list (see state.py). Each step applies one
masked action (actions.py) and returns a reward computed from the *current*
model's predicted hadron spectrum vs the target (reward every step, per the
addendum). The expensive spectrum call is memoized on a canonical model key, so
revisited models are free.

The env is deliberately decoupled from any RL algorithm: it emits a structured
observation {"model", "mask"} and a scalar reward. Featurization into tensors is
the policy's job (models/tokenizer.py), so PPO / GFlowNet / evolution can all
drive the same env.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import math

from .actions import ActionSpace
from .state import Model
from ..config import Config
from ..physics.groups import GroupLadder
from ..physics.spectrum import canonical_key, spectrum_of
from ..reward import make_metric, build_target


class SMModelEnv:
    def __init__(self, engine, cfg: Config | None = None,
                 target: Optional[np.ndarray] = None,
                 target_quarks: tuple[str, ...] = ("u", "d", "s"),
                 cache: dict | None = None):
        self.engine = engine
        self.cfg = cfg or Config()
        self.ladder = GroupLadder(engine, self.cfg.groups)
        self.aspace = ActionSpace(self.cfg.env, self.ladder)
        self.metric = make_metric(self.cfg.reward, self.cfg.env.n_u1)
        # cache may be a plain dict (in-memory) or a SpectrumCache (persistent).
        self.cache = cache if cache is not None else {}

        if target is None:
            target, self._target_model = build_target(
                engine, self.ladder, self.cfg.env, quarks=target_quarks,
                cache=self.cache)
        else:
            self._target_model = None
        self.target = target

        self.model: Model | None = None
        self._prev_score = 0.0
        self._steps = 0
        self._mask = None
        # Persistent visit counts for the count-based novelty bonus (keyed on
        # canonical_key, i.e. per distinct physics). Global across episodes.
        self._visit_counts: dict = {}
        # Multiplier on novelty_coef, driven by the learner for annealing
        # (1.0 = full bonus, 0.0 = off). See PPO.train / novelty_anneal_iters.
        self._novelty_scale: float = 1.0

    # ---- gym-like API -----------------------------------------------------

    @property
    def num_actions(self) -> int:
        return len(self.aspace)

    def reset(self, start_group: str | None = None, seed_model: Model | None = None):
        if seed_model is not None:
            self.model = seed_model.copy()
        else:
            g = start_group or self.ladder.default.gm
            self.model = Model(group=g)
        self._steps = 0
        self._prev_score = self._score(self.model)
        self._mask = self.aspace.mask(self.model)
        return self._obs(), {"score": self._prev_score}

    def step(self, action: int):
        assert self.model is not None, "call reset() first"
        desc = self.aspace.actions[action]
        self.model = self.aspace.apply(self.model, action)
        self._steps += 1

        score, match = self._score(self.model, return_match=True)
        if self.cfg.env.reward_mode == "delta":
            reward = score - self._prev_score
        else:  # "absolute"
            reward = score
        self._prev_score = score

        # Count-based novelty (intrinsic) reward on the *landed* model's physics.
        # Decays as beta/sqrt(n), so revisiting a config earns steadily less and
        # the policy is pushed toward unexplored (here: new-charge) models.
        novelty = 0.0
        if self.cfg.env.novelty_coef > 0.0:
            key = canonical_key(self.model, self.ladder, self.cfg.env.cpt_auto_conjugate)
            n = self._visit_counts.get(key, 0) + 1
            self._visit_counts[key] = n
            coef = self.cfg.env.novelty_coef * self._novelty_scale
            if self.cfg.env.novelty_mode == "first_visit":
                novelty = coef if n == 1 else 0.0
            else:  # "count": decaying, paid every visit
                novelty = coef / math.sqrt(n)
            reward += novelty

        if self.cfg.env.allow_stop:
            terminated = (desc.kind == "stop")       # legacy: agent commits via STOP
        else:
            # STOP is masked away; the only voluntary end is solving the task.
            # Reaching an exact match becomes an absorbing goal that terminates
            # the episode (otherwise it runs to max_steps truncation).
            terminated = match.exact
        truncated = (self._steps >= self.cfg.env.max_steps)
        if terminated:
            mode = self.cfg.env.terminal_mode
            if mode == "exact":
                if match.exact:
                    reward += self.cfg.env.terminal_bonus
            elif mode == "graded":
                reward += self.cfg.env.terminal_bonus * score
            elif mode == "shaped":
                reward += self.cfg.env.terminal_bonus * score
                if match.exact:
                    reward += self.cfg.env.terminal_exact_bonus
            else:
                raise ValueError(f"unknown terminal_mode {mode!r}")

        self._mask = self.aspace.mask(self.model)
        info = {"score": score, "exact": match.exact, "action": desc.name,
                "novelty": novelty, **match.info}
        return self._obs(), float(reward), terminated, truncated, info

    # ---- helpers ----------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        return self._mask

    def _obs(self) -> dict:
        return {"model": self.model, "mask": self._mask}

    def _score(self, model: Model, return_match: bool = False):
        spec = spectrum_of(self.engine, self.ladder, model, self.cfg.env, cache=self.cache)
        match = self.metric.match(spec, self.target)
        return (match.score, match) if return_match else match.score
