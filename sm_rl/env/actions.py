"""The (flat, maskable) action space.

Actions are small deterministic edits to the model, per the addendum:
  STOP, CHANGE_TYPE(+/-), CHANGE_RANK(+/-), ADD, REMOVE_LAST, and MODIFY of the
  edit-target parton (spin toggle, rep +/-, each charge +/-). A movable cursor
  (CURSOR_PREV/NEXT) is added only when EnvConfig.use_cursor is set; otherwise
  the edit target is always the most-recently-added parton.

Every transition is pure Python (no engine call). The expensive spectrum
evaluation happens in the environment's reward, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import Model, Parton
from ..physics.groups import GroupLadder

# default parton created by ADD: fundamental rep, scalar (boson), zero charges
DEFAULT_REP_INDEX = 1
DEFAULT_SPIN = 1  # boson / spin 0


@dataclass(frozen=True)
class ActionDesc:
    name: str
    kind: str
    param: object = None


class ActionSpace:
    def __init__(self, env_cfg, ladder: GroupLadder):
        self.cfg = env_cfg
        self.ladder = ladder
        self.actions: list[ActionDesc] = self._build()
        self.index = {a.name: i for i, a in enumerate(self.actions)}

    def _build(self) -> list[ActionDesc]:
        A = [
            ActionDesc("STOP", "stop"),
            ActionDesc("TYPE_UP", "type", +1),
            ActionDesc("TYPE_DOWN", "type", -1),
            ActionDesc("RANK_UP", "rank", +1),
            ActionDesc("RANK_DOWN", "rank", -1),
            ActionDesc("ADD", "add"),
            ActionDesc("REMOVE_LAST", "remove"),
            ActionDesc("SPIN_TOGGLE", "spin"),
            ActionDesc("REP_UP", "rep", +1),
            ActionDesc("REP_DOWN", "rep", -1),
        ]
        for c in range(self.cfg.n_u1):
            A.append(ActionDesc(f"CHARGE{c}_UP", "charge", (c, +1)))
            A.append(ActionDesc(f"CHARGE{c}_DOWN", "charge", (c, -1)))
        if self.cfg.use_cursor:
            A.append(ActionDesc("CURSOR_PREV", "cursor", -1))
            A.append(ActionDesc("CURSOR_NEXT", "cursor", +1))
        return A

    def __len__(self) -> int:
        return len(self.actions)

    # ---- transition -------------------------------------------------------

    def apply(self, model: Model, a: int) -> Model:
        """Return a new model with action `a` applied (input is not mutated)."""
        m = model.copy()
        desc = self.actions[a]
        kind, param = desc.kind, desc.param

        if kind == "stop":
            return m
        if kind == "type":
            new_spec = self.ladder.type_step(self.ladder.spec(m.group), param)
            self._reproject(m, new_spec.gm)
        elif kind == "rank":
            nxt = self.ladder.rank_step(self.ladder.spec(m.group), param)
            if nxt is not None:
                self._reproject(m, nxt.gm)
        elif kind == "add":
            if m.n < self.cfg.max_partons:
                spec = self.ladder.spec(m.group)
                m.partons.append(Parton(
                    rep_index=self.ladder.clamp_index(spec, DEFAULT_REP_INDEX),
                    spin=DEFAULT_SPIN,
                    charges=[0.0] * self.cfg.n_u1,
                    flavour=m.new_flavour(),
                ))
                m.cursor = m.n - 1
        elif kind == "remove":
            if m.n > 0:
                m.partons.pop()                    # remove most-recently-added
                m.cursor = min(m.cursor, m.n - 1)
        elif kind == "cursor":
            if m.n > 0:
                m.cursor = int(np.clip((m.cursor if m.cursor >= 0 else m.n - 1) + param, 0, m.n - 1))
        else:
            p = m.edit_target(self.cfg.use_cursor)
            if p is not None:
                self._edit_parton(m, p, kind, param)
        return m

    def _reproject(self, m: Model, new_group: str) -> None:
        m.group = new_group
        spec = self.ladder.spec(new_group)
        for p in m.partons:
            p.rep_index = self.ladder.clamp_index(spec, p.rep_index)  # round reps down

    def _edit_parton(self, m: Model, p: Parton, kind: str, param) -> None:
        spec = self.ladder.spec(m.group)
        if kind == "spin":
            p.spin = 1 if p.spin == 2 else 2
        elif kind == "rep":
            p.rep_index = self.ladder.clamp_index(spec, p.rep_index + param)
        elif kind == "charge":
            c, sign = param
            step = self.cfg.charge_steps[c]
            p.charges[c] = float(np.clip(p.charges[c] + sign * step,
                                         -self.cfg.charge_max, self.cfg.charge_max))

    # ---- masking ----------------------------------------------------------

    def mask(self, model: Model) -> np.ndarray:
        m = model
        spec = self.ladder.spec(m.group)
        has = m.n > 0
        p = m.edit_target(self.cfg.use_cursor)
        n_reps = self.ladder.n_reps(spec)
        out = np.zeros(len(self.actions), dtype=bool)
        for i, d in enumerate(self.actions):
            k, prm = d.kind, d.param
            if k == "stop":
                ok = m.n >= self.cfg.min_partons_before_stop
            elif k == "type":
                ok = True                                  # type_step cycles
            elif k == "rank":
                ok = self.ladder.rank_step(spec, prm) is not None
            elif k == "add":
                ok = m.n < self.cfg.max_partons
            elif k == "remove":
                ok = has
            elif k == "spin":
                ok = has
            elif k == "rep":
                ok = has and (0 <= p.rep_index + prm < n_reps)
            elif k == "charge":
                c, sign = prm
                ok = has and abs(p.charges[c] + sign * self.cfg.charge_steps[c]) <= self.cfg.charge_max + 1e-9
            elif k == "cursor":
                cur = p and self.cfg.use_cursor
                if not (has and self.cfg.use_cursor):
                    ok = False
                else:
                    nxt = int(np.clip((m.cursor if m.cursor >= 0 else m.n - 1) + prm, 0, m.n - 1))
                    ok = nxt != (m.cursor if m.cursor >= 0 else m.n - 1)
            else:
                ok = False
            out[i] = ok
        return out
