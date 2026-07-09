"""Smoke-test the Wolfram/GroupMath engine wrapper.

Run:  <python-with-wolframclient> scripts/engine_smoke.py
Confirms the kernel connects, GroupMath + spinSinglets load, and the core
primitives return the expected values for SU(3).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.physics import WolframEngine

FUND = (1, 0)      # SU(3) fundamental (3)
ANTIFUND = (0, 1)  # SU(3) antifundamental (3bar)


def main():
    eng = WolframEngine().start()
    try:
        reps = eng.reps_up_to_dim("SU3", 10)
        print("SU3 reps up to dim 10 (dimension-ordered):")
        for i, r in enumerate(reps):
            print(f"  r{i}: {r}  dim={eng.dim('SU3', r)}")

        print("\nfundamental:", FUND,
              "dim", eng.dim("SU3", FUND),
              "dynkin", eng.dynkin_index("SU3", FUND),
              "casimir", eng.casimir("SU3", FUND),
              "conj", eng.conjugate_rep("SU3", FUND),
              "cclass", eng.conjugacy_class("SU3", FUND))

        # meson: fund (x) antifund, both spin-1/2 fermions, distinct flavours.
        # Expect a spin-0 and a spin-1 singlet -> ((3,1),(1,1)) in any order.
        spins = eng.get_spins("SU3", [FUND, ANTIFUND], [2, 2], [1, 2])
        print("\nget_spins(SU3, [3, 3bar], fermions, distinct) =", spins,
              "  (expect spin-1 label 3 and spin-0 label 1)")

        # baryon: three fundamentals, all fermions, distinct flavours.
        spins3 = eng.get_spins("SU3", [FUND, FUND, FUND], [2, 2, 2], [1, 2, 3])
        print("get_spins(SU3, [3,3,3], fermions, distinct) =", spins3,
              "  (expect spin-3/2 label 4 and spin-1/2 label 2)")

        n = eng.min_copies_for_singlet("SU3", FUND)
        print("\nmin_copies_for_singlet(SU3, fund) =", n, " (expect 3)")
        print("\nENGINE SMOKE TEST: OK")
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
