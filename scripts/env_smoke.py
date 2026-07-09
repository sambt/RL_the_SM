"""End-to-end environment smoke test (no neural net).

  A. Random masked rollout -- exercises actions, masking, per-step reward.
  B. Scripted 'oracle' trajectory -- builds the exact (u,d,s) model through the
     action space and asserts the reward fires (exact match + terminal bonus).
     This proves the action space can express the target and the MDP is wired up.

Run:  ~/miniforge3/envs/jax/bin/python scripts/env_smoke.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # before numpy/scipy/torch
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.config import Config
from sm_rl.physics import WolframEngine
from sm_rl.env import SMModelEnv


def describe(model):
    parts = []
    for p in model.partons:
        parts.append(f"r{p.rep_index}/s{p.spin}/q{[round(c,2) for c in p.charges]}")
    return f"{model.group}[" + ", ".join(parts) + "]"


def main():
    cfg = Config()
    eng = WolframEngine(cfg.engine).start()
    try:
        env = SMModelEnv(eng, cfg)
        print(f"num_actions = {env.num_actions}")
        print("actions:", [a.name for a in env.aspace.actions])
        print(f"target: {len(env.target)} states\n")

        # --- A. random masked rollout ---------------------------------
        print("=" * 64, "\n[A] random masked rollout (start SU3)")
        rng = np.random.default_rng(0)
        obs, info = env.reset(start_group="SU3")
        print(f"  reset score={info['score']:+.1f}")
        for t in range(12):
            mask = env.action_mask()
            a = int(rng.choice(np.flatnonzero(mask)))
            obs, r, term, trunc, info = env.step(a)
            print(f"  t{t:2d} {info['action']:<12} r={r:+6.1f} score={info['score']:+6.1f} "
                  f"miss={info['missing']:>3} extra={info['extra']:>3}  {describe(obs['model'])}")
            if term or trunc:
                break

        # --- B. scripted oracle trajectory ----------------------------
        print("=" * 64, "\n[B] scripted oracle: build exact (u,d,s)")
        idx = env.aspace.index

        def do(name):
            return env.step(idx[name])

        env.reset(start_group="SU3")
        # u = fund fermion [EM=2, B=1, S=0, I3=1/2]
        do("ADD"); do("SPIN_TOGGLE"); do("CHARGE0_UP"); do("CHARGE0_UP"); do("CHARGE1_UP"); do("CHARGE3_UP")
        # d = fund fermion [EM=-1, B=1, S=0, I3=-1/2]
        do("ADD"); do("SPIN_TOGGLE"); do("CHARGE0_DOWN"); do("CHARGE1_UP"); do("CHARGE3_DOWN")
        # s = fund fermion [EM=-1, B=1, S=-1, I3=0]
        do("ADD"); do("SPIN_TOGGLE"); do("CHARGE0_DOWN"); do("CHARGE1_UP"); do("CHARGE2_DOWN")
        obs, r, term, trunc, info = do("STOP")
        print(f"  built {describe(obs['model'])}")
        print(f"  STOP reward={r:+.1f}  exact={info['exact']}  miss={info['missing']} extra={info['extra']}")
        assert info["exact"], "scripted oracle did not reach an exact match!"
        assert term and r >= cfg.env.terminal_bonus, "terminal bonus not applied on exact STOP"
        print("=" * 64, "\nENV SMOKE TEST: OK")
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
