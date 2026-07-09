"""Phase 0: validate the physics + reward core against ground truth.

Checks, in order:
  1. CPT auto-conjugation on [u,d] reproduces the notebook's explicit
     [u, d, u^c, d^c] spectrum (same multiset of hadron signatures).
  2. Build the (u,d,s) target spectrum and summarize it by spin sector.
  3. The count reward makes the true (u,d,s) model the unique argmax:
     - reward(true) == 0 (exact),
     - dropping / perturbing a quark gives a strictly worse (negative) score.

Run:  ~/miniforge3/envs/jax/bin/python scripts/phase0_validate.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # before numpy/scipy/torch
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.config import Config
from sm_rl.physics import WolframEngine
from sm_rl.physics.groups import GroupLadder
from sm_rl.physics.spectrum import spectrum_of
from sm_rl.env.state import Model, Parton
from sm_rl.reward import make_metric, build_target
from sm_rl.reward.target import QUARK_CHARGES, fundamental_index, FERMION

SPIN_NAME = {1.0: "0", 2.0: "1/2", 3.0: "1", 4.0: "3/2"}


def sig_multiset(spec):
    return Counter(tuple(np.round(r, 6)) for r in spec)


def summarize(spec, n_u1=4):
    print(f"  {len(spec)} states")
    by_spin = Counter(r[n_u1] for r in spec)
    for s in sorted(by_spin):
        print(f"    spin {SPIN_NAME.get(s, s):>4}: {by_spin[s]} states")


def main():
    cfg = Config()
    eng = WolframEngine(cfg.engine).start()
    try:
        ladder = GroupLadder(eng, cfg.groups)
        spec_su3 = ladder.spec("SU3")
        fund = fundamental_index(ladder, "SU3")
        anti = 2  # (0,1) antifundamental, dimension-order index 2
        cache = {}

        # --- 1. CPT([u,d]) == explicit [u,d,u^c,d^c] ----------------------
        print("=" * 64)
        print("[1] CPT auto-conjugation vs explicit conjugate fields (2-flavour)")
        env_cpt = cfg.env
        ud_target, _ = build_target(eng, ladder, env_cpt, quarks=("u", "d"), cache=cache)

        env_expl = Config().env
        env_expl.cpt_auto_conjugate = False
        m = Model("SU3")
        for q, ridx in [("u", fund), ("d", fund)]:
            m.partons.append(Parton(ridx, FERMION, list(QUARK_CHARGES[q]), m.new_flavour()))
        for q, ridx in [("u", anti), ("d", anti)]:  # explicit conjugates
            m.partons.append(Parton(ridx, FERMION, [-c for c in QUARK_CHARGES[q]], m.new_flavour()))
        ud_explicit = spectrum_of(eng, ladder, m, env_expl, cache={})

        same = sig_multiset(ud_target) == sig_multiset(ud_explicit)
        print(f"  CPT[u,d]:      {len(ud_target)} states")
        print(f"  explicit uddc: {len(ud_explicit)} states")
        print(f"  MULTISETS EQUAL: {same}")
        assert same, "CPT does not reproduce explicit conjugate spectrum!"

        # --- 2. (u,d,s) target -------------------------------------------
        print("=" * 64)
        print("[2] (u,d,s) target spectrum")
        target, target_model = build_target(eng, ladder, cfg.env, quarks=("u", "d", "s"), cache=cache)
        summarize(target)

        # --- 3. count reward: true model is argmax -----------------------
        print("=" * 64)
        print("[3] count reward (true model must be the unique argmax)")
        metric = make_metric(cfg.reward, cfg.env.n_u1)

        r_true = metric.match(target, target)
        print(f"  reward(true u,d,s)          score={r_true.score:+.1f}  exact={r_true.exact}  {r_true.info}")
        assert r_true.exact and r_true.score == 0.0

        # perturbation A: drop the strange quark -> (u,d) only
        pert_ud, _ = build_target(eng, ladder, cfg.env, quarks=("u", "d"), cache=cache)
        r_ud = metric.match(pert_ud, target)
        print(f"  reward(drop s -> u,d)       score={r_ud.score:+.1f}  exact={r_ud.exact}  {r_ud.info}")

        # perturbation B: wrong strange charge (shift s EM by +3 units)
        mB = target_model.copy()
        mB.partons[2].charges[0] += 3.0
        specB = spectrum_of(eng, ladder, mB, cfg.env, cache={})
        r_B = metric.match(specB, target)
        print(f"  reward(s EM shifted)        score={r_B.score:+.1f}  exact={r_B.exact}  {r_B.info}")

        # perturbation C: extra spurious quark (a second u')
        mC = target_model.copy()
        mC.partons.append(Parton(fund, FERMION, list(QUARK_CHARGES["u"]), mC.new_flavour()))
        specC = spectrum_of(eng, ladder, mC, cfg.env, cache={})
        r_C = metric.match(specC, target)
        print(f"  reward(add spurious u')     score={r_C.score:+.1f}  exact={r_C.exact}  {r_C.info}")

        ok = (r_true.score == 0.0) and all(r.score < 0 for r in [r_ud, r_B, r_C])
        print("=" * 64)
        print("PHASE 0 VALIDATION:", "OK" if ok else "FAILED")
        assert ok
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
