"""Offline diagnostic: is the f1nostop 0.786 plateau a reward-landscape local
optimum, or an exploration failure? The plateau model is structurally the SM
(SU3, 3 fundamental fermions) -- only the U(1) charges are wrong. So we probe
charge space around it, using the SAME physics (spectrum_of + canonical_key) and
the SAME metric (f1 + physical) as training.

Two tests:
  (A) best-first search over the +/-1 charge lattice from the plateau: does ANY
      monotone-improving path reach a higher score / exact match nearby?
  (B) direct lattice walk from the plateau charges toward the KNOWN u,d,s target
      (partons matched by min charge-distance): is that specific route a smooth
      climb, or is there a valley between the plateau and the SM?

If (A)/(B) reach ~1.0 with no deep valley -> exploration failure (PPO knobs help).
If the plateau is a strict local max walled below 1.0 -> landscape/reward trap.
"""
import heapq
import itertools
import json
import os
import pickle

from sm_rl.config import Config
from sm_rl.env.state import Model, Parton
from sm_rl.env.sm_env import SMModelEnv
from sm_rl.physics.engine import WolframEngine

RUN = "runs/f1nostop_20260730_151652"


def build_env():
    cfg = Config()
    cfg.reward.metric = "f1"
    cfg.reward.normalization = "physical"
    eng = WolframEngine(cfg.engine).start()
    # Load the warm shared cache into a PLAIN dict so new computes stay in memory
    # and never flush back to the shared pickle.
    cache = {}
    cpath = os.environ.get("SM_RL_CACHE")
    if cpath and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            cache.update(pickle.load(f))
        print(f"[cache] loaded {len(cache):,} entries from {cpath}", flush=True)
    env = SMModelEnv(eng, cfg, cache=cache)
    return env, cfg


def model_from_charges(group, charge_lists, spin=2, rep_index=1):
    m = Model(group=group)
    for ch in charge_lists:
        m.partons.append(Parton(rep_index=rep_index, spin=spin,
                                charges=list(ch), flavour=m.new_flavour()))
    return m


def charges_of(model):
    return tuple(tuple(p.charges) for p in model.partons)


def score(env, model):
    s, match = env._score(model, return_match=True)
    return s, match.exact


def neighbors(env, cfg, charge_lists):
    """All +/-1 lattice moves on every (parton, charge), within charge_max."""
    steps = cfg.env.charge_steps
    cmax = cfg.env.charge_max
    out = []
    cl = [list(c) for c in charge_lists]
    for i in range(len(cl)):
        for c in range(cfg.env.n_u1):
            for sign in (+1, -1):
                v = cl[i][c] + sign * steps[c]
                if abs(v) > cmax + 1e-9:
                    continue
                nc = [list(x) for x in cl]
                nc[i][c] = v
                out.append(tuple(tuple(x) for x in nc))
    return out


def test_A_bestfirst(env, cfg, group, start_charges, start_score, budget=1500):
    """Best-first search: expand the highest-scoring frontier node's neighbors."""
    seen = {start_charges: start_score}
    heap = [(-start_score, start_charges)]
    best = (start_score, start_charges)
    exact_found = None
    evals = 0
    while heap and evals < budget:
        negs, ch = heapq.heappop(heap)
        for nc in neighbors(env, cfg, ch):
            if nc in seen:
                continue
            s, ex = score(env, model_from_charges(group, nc))
            evals += 1
            seen[nc] = s
            heapq.heappush(heap, (-s, nc))
            if s > best[0]:
                best = (s, nc)
            if ex and exact_found is None:
                exact_found = nc
            if evals >= budget:
                break
    strict_local_max = best[0] <= start_score + 1e-9
    return dict(evals=evals, best_score=best[0], best_charges=best[1],
                exact_found=exact_found is not None, exact_charges=exact_found,
                strict_local_max=strict_local_max, n_seen=len(seen))


def match_partons(start_charges, target_charges):
    """Greedy min-L1 assignment of start partons to target partons."""
    n = len(target_charges)
    used = set()
    order = []
    for sc in start_charges:
        best_j, best_d = None, None
        for j in range(n):
            if j in used:
                continue
            d = sum(abs(a - b) for a, b in zip(sc, target_charges[j]))
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        order.append(best_j)
    return order


def test_B_path(env, cfg, group, start_charges, target_charges):
    """Walk each charge from start toward its matched target, one lattice step at
    a time (greedy coordinate move that most reduces remaining distance), scoring
    every intermediate. Reveals any valley between plateau and the SM."""
    order = match_partons(start_charges, target_charges)
    tgt = [target_charges[j] for j in order]
    steps = cfg.env.charge_steps
    cur = [list(c) for c in start_charges]
    traj = []
    s0, ex0 = score(env, model_from_charges(group, cur))
    traj.append((s0, ex0))
    for _ in range(400):  # generous cap
        # remaining lattice moves needed
        moves = []
        for i in range(len(cur)):
            for c in range(cfg.env.n_u1):
                diff = tgt[i][c] - cur[i][c]
                if abs(diff) > 1e-9:
                    sign = 1 if diff > 0 else -1
                    moves.append((i, c, sign))
        if not moves:
            break
        # pick the single move that yields the highest score (greedy along the
        # monotone route toward target) -- this finds the *best* monotone path.
        best_mv, best_s, best_ex = None, None, None
        for (i, c, sign) in moves:
            nc = [list(x) for x in cur]
            nc[i][c] = nc[i][c] + sign * steps[c]
            s, ex = score(env, model_from_charges(group, nc))
            if best_s is None or s > best_s:
                best_s, best_ex, best_mv = s, ex, (i, c, sign, nc)
        i, c, sign, nc = best_mv
        cur = nc
        traj.append((best_s, best_ex))
        if best_ex:
            break
    scores = [t[0] for t in traj]
    reached_exact = any(t[1] for t in traj)
    valley = min(scores) < scores[0] - 1e-9
    return dict(n_steps=len(traj) - 1, start=scores[0], end=scores[-1],
                min_along_path=min(scores), max_along_path=max(scores),
                reached_exact=reached_exact, valley=valley, scores=scores)


def main():
    env, cfg = build_env()
    best = json.load(open(f"{RUN}/best_model.json"))["model"]
    group = best["group"]
    start_charges = tuple(tuple(p["charges"]) for p in best["partons"])
    m0 = model_from_charges(group, start_charges,
                            spin=best["partons"][0]["spin"],
                            rep_index=best["partons"][0]["rep_index"])
    s0, ex0 = score(env, m0)
    print(f"\n[plateau] group={group} score={s0:.4f} exact={ex0}")
    print(f"[plateau] charges={start_charges}")

    tm = env._target_model
    tgt_charges = tuple(tuple(p.charges) for p in tm.partons)
    st, ext = score(env, tm)
    print(f"[target ] score={st:.4f} exact={ext}  charges={tgt_charges}")

    print("\n=== TEST A: best-first search over charge lattice ===", flush=True)
    A = test_A_bestfirst(env, cfg, group, start_charges, s0)
    print(json.dumps(A, indent=2, default=str))

    print("\n=== TEST B: direct path plateau -> known u,d,s target ===", flush=True)
    B = test_B_path(env, cfg, group, start_charges, tgt_charges)
    print(json.dumps({k: (v if k != "scores" else [round(x, 3) for x in v])
                      for k, v in B.items()}, indent=2))

    # Verdict
    print("\n=== VERDICT ===")
    if A["exact_found"] or B["reached_exact"]:
        print("EXPLORATION FAILURE: a monotone/near-monotone charge path from the")
        print("plateau reaches an EXACT match. The landscape is fine; PPO simply")
        print("did not explore charge space enough. -> optimization knobs should help.")
    elif A["strict_local_max"]:
        print("LANDSCAPE TRAP: the plateau is a strict local max over the +/-1 charge")
        print("lattice (no neighbor scores higher within the search budget).")
        print("-> exploration knobs will NOT help; the reward metric needs work.")
    else:
        print("MIXED: improving neighbors exist but no exact match found within budget.")
        print(f"  best reachable={A['best_score']:.4f}; path valley={B['valley']}.")

    out = f"{RUN}/charge_landscape.json"
    json.dump({"plateau_score": s0, "target_score": st, "A": A, "B": B},
              open(out, "w"), indent=2, default=str)
    print(f"\nsaved {out}")
    env.engine.stop()


if __name__ == "__main__":
    main()
