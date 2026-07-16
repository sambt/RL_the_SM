"""Mathematica-free unit tests for the pure-Python logic.

Run:  ~/miniforge3/envs/jax/bin/python tests/test_core.py     (or via pytest)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # before numpy/scipy
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_engine import MockEngine
from sm_rl.config import Config
from sm_rl.physics.groups import GroupLadder
from sm_rl.physics.spectrum import normalize_spectrum, resolve_fields, canonical_key
from sm_rl.env.state import Model, Parton
from sm_rl.env.actions import ActionSpace
from sm_rl.reward.count import CountReward
from sm_rl.reward.f1 import F1Reward
from sm_rl.reward.hungarian import HungarianReward
from sm_rl.algos.base import compute_gae


def test_normalization_gcd():
    # column of {2,-4,6} -> gcd 2 -> {1,-2,3}; spin column untouched.
    spec = np.array([[2.0, 0.0, 0.0, 0.0, 1.0],
                     [-4.0, 0.0, 0.0, 0.0, 3.0],
                     [6.0, 0.0, 0.0, 0.0, 2.0]])
    out = normalize_spectrum(spec, n_u1=4, mode="gcd")
    assert np.allclose(out[:, 0], [1.0, -2.0, 3.0])
    assert np.allclose(out[:, 4], [1.0, 3.0, 2.0])   # spin label preserved
    print("ok test_normalization_gcd")


def test_count_reward():
    cfg = Config().reward
    metric = CountReward(cfg, n_u1=4)
    target = np.array([[1, 0, 0, 0, 1.0], [0, 0, 0, 0, 2.0]])
    # exact
    r = metric.match(target.copy(), target)
    assert r.exact and r.score == 0.0
    # one missing
    r = metric.match(target[:1], target)
    assert r.info["missing"] == 1 and r.info["extra"] == 0 and r.score == -1.0
    # one extra
    extra = np.vstack([target, [[9, 0, 0, 0, 1.0]]])
    r = metric.match(extra, target)
    assert r.info["extra"] == 1 and r.info["missing"] == 0
    print("ok test_count_reward")


def test_hungarian_reward():
    cfg = Config().reward
    metric = HungarianReward(cfg, n_u1=4)
    target = np.array([[1, 0, 0, 0, 1.0], [0, 0, 0, 0, 2.0]])
    r = metric.match(target.copy(), target)
    assert r.exact and abs(r.score - 1.0) < 1e-6
    r = metric.match(target[:1], target)          # one missing -> lower score
    assert r.info["missing"] == 1 and r.score < 1.0
    print("ok test_hungarian_reward")


def _f1_cfg():
    cfg = Config().reward
    cfg.metric, cfg.normalization = "f1", "physical"
    return cfg


def _fake_target(n=12):
    """A stand-in target in the real charge convention: EM/B/S are physical x3,
    so the charge quantum is 3 (this is what makes normalization matter)."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        rows.append([rng.choice([-6., -3., 0., 3., 6.]), rng.choice([-3., 0., 3.]),
                     rng.choice([-3., 0., 3.]), rng.choice([-1., 0., 1.]),
                     float(rng.choice([1., 2., 3.]))])
    return np.unique(np.array(rows), axis=0)


def test_normalization_physical():
    # EM/B/S divided by 3, I3 by 1; spin label untouched.
    spec = np.array([[6.0, 3.0, -3.0, 1.5, 2.0]])
    out = normalize_spectrum(spec, n_u1=4, mode="physical", scale=(3.0, 3.0, 3.0, 1.0))
    assert np.allclose(out[0], [2.0, 1.0, -1.0, 1.5, 2.0])
    print("ok test_normalization_physical")


def test_f1_target_is_optimal():
    """The target must be the unique global maximum: score 1.0, and strictly
    better than dropping a state, adding a spurious one, or shifting a charge."""
    t = _fake_target()
    m = F1Reward(_f1_cfg(), n_u1=4)
    r = m.match(t.copy(), t)
    assert r.exact and abs(r.score - 1.0) < 1e-9, r
    assert m.match(t[:-1].copy(), t).score < 1.0            # missing a state
    extra = np.vstack([t, [[9.0, 3.0, 3.0, 1.0, 1.0]]])
    assert m.match(extra, t).score < 1.0                    # spurious state
    shifted = t.copy(); shifted[:, 0] += 3.0                # one charge quantum off
    assert m.match(shifted, t).score < 1.0
    print("ok test_f1_target_is_optimal")


def test_f1_predicting_more_beats_predicting_nothing():
    """REGRESSION: the Hungarian reward's spurious penalty is additive and
    unbounded while its matched count saturates, so its score decreases ~linearly
    in n_pred -- making the empty model near-optimal and punishing SU(3) (the only
    group that forms many singlets) hardest. A usable reward must never make
    'predict nothing' beat a partially-correct prediction."""
    t = _fake_target()
    m = F1Reward(_f1_cfg(), n_u1=4)
    empty = m.match(np.zeros((0, 5)), t).score
    half = m.match(t[: len(t) // 2].copy(), t).score        # half the target, all correct
    assert empty == 0.0
    assert half > empty, f"half-correct ({half}) must beat empty ({empty})"
    # and richer-but-imperfect must still beat nothing
    noisy = np.vstack([t, t[:3] + np.array([3.0, 0, 0, 0, 0])])
    assert m.match(noisy, t).score > empty
    print("ok test_f1_predicting_more_beats_predicting_nothing")


def test_f1_bounded_under_overproduction():
    """Spam must be penalised but bounded: score stays in [0, 1] and decreases
    toward 0 rather than diverging to -inf (which is what makes the empty model
    the global optimum under an additive spurious penalty)."""
    t = _fake_target()
    m = F1Reward(_f1_cfg(), n_u1=4)
    prev = 1.0
    for k in (0, 10, 50, 200):
        junk = np.tile([[9.0, 3.0, 3.0, 1.0, 1.0]], (k, 1))
        s = m.match(np.vstack([t, junk]) if k else t.copy(), t).score
        assert 0.0 <= s <= 1.0, f"score {s} out of bounds at k={k}"
        assert s <= prev + 1e-9, "overproduction must not increase the score"
        prev = s
    assert prev < 0.5, "massive overproduction must be clearly penalised"
    print("ok test_f1_bounded_under_overproduction")


def test_f1_partial_credit_needs_physical_normalization():
    """REGRESSION: with normalization='none' the charge quantum is 3, which is
    >= match_radius (3.0), so a one-unit charge error earns ZERO partial credit
    and the reward carries no gradient toward the right charges."""
    # single isolated state, so nothing can accidentally match a *different* row
    t = np.array([[3.0, 3.0, 0.0, 0.0, 2.0]])
    near = np.array([[6.0, 3.0, 0.0, 0.0, 2.0]])     # one physical unit of EM away
    cfg_raw = Config().reward; cfg_raw.metric, cfg_raw.normalization = "f1", "none"
    raw = F1Reward(cfg_raw, n_u1=4).match(near.copy(), t).score
    graded = F1Reward(_f1_cfg(), n_u1=4).match(near.copy(), t).score
    assert raw == 0.0, ("raw charges put the quantum (3) at/beyond match_radius (3.0), "
                        "so a one-unit miss earns nothing -- no gradient")
    assert graded > 0.0, "physical normalization must give near-misses partial credit"
    assert graded > raw

    # and it must be graded, not a cliff: further away => strictly less credit
    far = np.array([[9.0, 3.0, 0.0, 0.0, 2.0]])      # two units away
    m = F1Reward(_f1_cfg(), n_u1=4)
    assert m.match(t.copy(), t).score > m.match(near, t).score > m.match(far, t).score >= 0.0
    print("ok test_f1_partial_credit_needs_physical_normalization")


def test_gae():
    # constant reward 1, zero values, no done, gamma=1, lambda=1 over 3 steps.
    adv, ret = compute_gae([1, 1, 1], [0, 0, 0], [False, False, False],
                           last_value=0.0, gamma=1.0, lam=1.0)
    assert np.allclose(adv, [3, 2, 1])
    assert np.allclose(ret, [3, 2, 1])
    print("ok test_gae")


def _ladder():
    return GroupLadder(MockEngine().start(), Config().groups)


def test_actions_apply_and_mask():
    cfg = Config()
    ladder = _ladder()
    aspace = ActionSpace(cfg.env, ladder)
    idx = aspace.index
    m = Model("SU3")

    # empty model: cannot REMOVE / edit; can ADD / change group. STOP is masked
    # until the model has >= min_partons_before_stop partons (default 2).
    mask = aspace.mask(m)
    assert not mask[idx["REMOVE_LAST"]] and not mask[idx["SPIN_TOGGLE"]]
    assert mask[idx["ADD"]]
    assert mask[idx["STOP"]] == (0 >= cfg.env.min_partons_before_stop)

    m = aspace.apply(m, idx["ADD"])
    assert m.n == 1 and m.partons[0].rep_index == 1 and m.partons[0].spin == 1  # fundamental boson
    # one parton is still below the STOP threshold (default 2)
    assert aspace.mask(m)[idx["STOP"]] == (1 >= cfg.env.min_partons_before_stop)
    m2 = aspace.apply(m, idx["ADD"])
    assert aspace.mask(m2)[idx["STOP"]] == (2 >= cfg.env.min_partons_before_stop)
    m = aspace.apply(m, idx["SPIN_TOGGLE"])
    assert m.partons[0].spin == 2
    m = aspace.apply(m, idx["CHARGE0_UP"])
    assert m.partons[0].charges[0] == cfg.env.charge_steps[0]
    m = aspace.apply(m, idx["REMOVE_LAST"])
    assert m.n == 0
    print("ok test_actions_apply_and_mask")


def test_group_change_clamps_rep_index():
    ladder = _ladder()
    cfg = Config()
    aspace = ActionSpace(cfg.env, ladder)
    idx = aspace.index
    m = Model("SU3")
    m = aspace.apply(m, idx["ADD"])
    m.partons[0].rep_index = 5           # (1,1) adjoint, dim 8, only valid in SU3
    m = aspace.apply(m, idx["RANK_DOWN"])  # SU3 -> SU2 (only 4 reps)
    assert m.group == "SU2"
    assert 0 <= m.partons[0].rep_index < ladder.n_reps(ladder.spec("SU2"))
    print("ok test_group_change_clamps_rep_index")


def test_cpt_resolve_and_canonical():
    ladder = _ladder()
    m = Model("SU3")
    m.partons.append(Parton(1, 2, [2.0, 1.0, 0.0, 0.5], m.new_flavour()))  # a 'u'
    fields = resolve_fields(m, ladder, cpt_auto_conjugate=True)
    assert len(fields) == 2                      # u + ubar
    reps = {f[0] for f in fields}
    assert (1, 0) in reps and (0, 1) in reps      # fundamental + antifundamental
    # canonical key is order-invariant
    k1 = canonical_key(m, ladder, True)
    m2 = m.copy(); m2.partons.reverse()
    assert k1 == canonical_key(m2, ladder, True)
    print("ok test_cpt_resolve_and_canonical")


def test_model_roundtrip():
    m = Model("SU3")
    m.partons.append(Parton(1, 2, [2.0, 1.0, 0.0, 0.5], m.new_flavour()))
    m2 = Model.from_dict(m.to_dict())
    assert m2.group == "SU3" and m2.n == 1
    p = m2.partons[0]
    assert p.rep_index == 1 and p.spin == 2 and p.charges == [2.0, 1.0, 0.0, 0.5]
    assert m2.describe() == m.describe()
    print("ok test_model_roundtrip")


def test_spectrum_cache_persist():
    import tempfile
    from sm_rl.physics.cache import SpectrumCache
    path = os.path.join(tempfile.mkdtemp(), "c.pkl")
    c = SpectrumCache(path, flush_every=1)
    key = ("SU3", ((1, 0), 2, (2.0, 1.0, 0.0, 0.5)))
    c[key] = np.array([[1.0, 2.0, 3.0]])
    assert key in c and c.n_writes == 1
    c2 = SpectrumCache(path)                       # reload from disk
    assert key in c2 and np.allclose(c2[key], [[1.0, 2.0, 3.0]])
    print("ok test_spectrum_cache_persist")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} CORE TESTS PASSED")


if __name__ == "__main__":
    main()
