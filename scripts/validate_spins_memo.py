"""Validate the canonicalized, persisted GetSpins memo against the real engine.

Checks, with live GroupMath calls:
  1. GetSpins is invariant under permuting the parton list and relabelling
     flavours (the two symmetries canonical_spins_args assumes).
  2. Full spectra computed through the memoized/canonical path equal those
     computed with raw, un-canonicalized GetSpins calls, for random models
     across several groups (incl. repeated fields and CPT conjugates).
  3. save_memo/load_memo round-trips and reproduces the same spectra with
     ZERO new GetSpins calls.

Run:  module load mathematica/15.0-fasrc01; uv run python scripts/validate_spins_memo.py
"""
import os, sys, random, tempfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sm_rl.config import Config
from sm_rl.physics import WolframEngine
from sm_rl.physics.groups import GroupLadder
from sm_rl.physics.spectrum import resolve_fields, compute_spectrum
from sm_rl.env.state import Model, Parton


def raw_spectrum(eng, group, fields, d_max, n_u1):
    """compute_spectrum, but through get_spins_raw (no memo, no canonicalization)."""
    from itertools import combinations_with_replacement
    n = len(fields)
    reps = [f[0] for f in fields]; spins = [f[1] for f in fields]
    charges = [np.asarray(f[2], dtype=float) for f in fields]
    flavours = list(range(1, n + 1)); rows = []
    for d in range(1, d_max + 1):
        for combo in combinations_with_replacement(range(n), d):
            idx = list(combo)
            inv = eng.get_spins_raw(group, [reps[i] for i in idx], [spins[i] for i in idx], [flavours[i] for i in idx])
            if inv:
                q = np.sum([charges[i] for i in idx], axis=0)
                for label, mult in inv:
                    rows.extend(np.append(q, label) for _ in range(mult))
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, n_u1 + 1))


def sorted_rows(a):
    return np.array(sorted(map(tuple, a.tolist()))) if len(a) else a


def main():
    random.seed(0)
    cfg = Config()
    eng = WolframEngine(cfg.engine).start()
    ladder = GroupLadder(eng, cfg.groups)
    t0 = time.time()

    # 1. symmetry of GetSpins itself
    print("[1] permutation / relabel invariance of GetSpins")
    cases = [("SU3", [(1,0),(0,1)], [2,2], [1,2]), ("SU3", [(1,0),(1,0),(1,0)], [2,2,2], [1,2,3]),
             ("SU3", [(1,0),(1,0),(1,0)], [2,2,2], [1,1,2]), ("SU2", [(1,),(1,)], [2,2], [1,1]),
             ("SU3", [(1,0),(1,0),(0,1)], [1,2,2], [4,4,7]), ("SO5", [(1,0),(1,0)], [2,2], [1,2]),
             ("G2", [(1,0),(1,0),(1,0)], [2,2,2], [1,2,3])]
    n_checks = 0
    for group, reps, spins, flavs in cases:
        ref = eng.get_spins_raw(group, reps, spins, flavs)
        idxs = list(range(len(reps)))
        for _ in range(4):
            perm = idxs[:]; random.shuffle(perm)
            relabel = {f: f + random.randint(0, 20) * 0 + 100 * random.randint(1, 3) for f in set(flavs)}
            out = eng.get_spins_raw(group, [reps[i] for i in perm], [spins[i] for i in perm],
                                    [relabel[flavs[i]] for i in perm])
            assert sorted(out) == sorted(ref), (group, reps, spins, flavs, perm, ref, out)
            n_checks += 1
    print(f"    ok ({n_checks} permuted/relabelled queries agree)")

    # 2. random models: memoized/canonical vs raw
    print("[2] random-model spectra: canonical memo path == raw path")
    groups = ["SU2", "SU3", "SU4", "SO5", "SP4", "G2"]
    n_models = 0
    for gi, group in enumerate(groups):
        spec = ladder.spec(group); n_reps = min(ladder.n_reps(spec), 3)
        for _ in range(4):
            m = Model(group)
            for _ in range(random.randint(1, 4)):
                m.partons.append(Parton(random.randrange(n_reps), random.choice([1, 2]),
                                        [float(random.choice([-2, -1, 0, 1, 2])) for _ in range(4)], m.new_flavour()))
            fields = resolve_fields(m, ladder, True)
            a = compute_spectrum(eng, group, fields, cfg.env.d_max, cfg.env.n_u1)
            b = raw_spectrum(eng, group, fields, cfg.env.d_max, cfg.env.n_u1)
            assert a.shape == b.shape and np.allclose(sorted_rows(a), sorted_rows(b)), (m.describe(), a, b)
            n_models += 1
        print(f"    {group}: ok  (memo: {eng.memo_stats()})")
    print(f"    ok ({n_models} models)")

    # 3. persistence round-trip with zero new calls
    print("[3] save_memo/load_memo round-trip")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memo.pkl")
        n = eng.save_memo(path)
        eng2 = WolframEngine(cfg.engine).start()
        assert eng2.load_memo(path) == n
        m = Model("SU3")
        for q in ([2.,1.,0.,.5], [-1.,1.,0.,-.5], [-1.,1.,-1.,0.]):
            m.partons.append(Parton(1, 2, q, m.new_flavour()))
        f = resolve_fields(m, ladder, True)
        a = compute_spectrum(eng, "SU3", f, 3, 4); b = compute_spectrum(eng2, "SU3", f, 3, 4)
        assert eng2.n_spins_calls == 0, eng2.n_spins_calls
        assert np.allclose(sorted_rows(a), sorted_rows(b))
        eng2.stop()
        print(f"    ok ({n} entries, {os.path.getsize(path)/1e3:.1f} KB, 0 new GetSpins calls on reload)")
    eng.stop()
    print(f"ALL SPINS-MEMO CHECKS PASSED in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
