"""Offline evaluation of a trained policy with the REAL engine (cheap now that
the GetSpins memo is persisted): what does the policy actually build?

Loads <run_dir>/config.json + a checkpoint, rebuilds env + policy exactly as
trained, runs N sampled episodes (and one greedy episode per start group), and
reports: final-model group / parton-count / spin histograms, score stats per
group, action-KIND usage, and the policy entropy split into "which kind" vs
"which value within charge_set". Prints the best model seen.

  uv run --extra rl python scripts/eval_policy.py runs/f1set_20260908_130441 \
      --ckpt final.pt --episodes 300
"""
import os, sys, json, argparse, collections
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sm_rl.config import Config
from sm_rl.physics import WolframEngine, SpectrumCache
from sm_rl.env import SMModelEnv
from sm_rl.models import Tokenizer, ActorCritic, GROUP_FEAT
from sm_rl.models.transformer import to_torch


def load_cfg(run_dir):
    c = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = Config()
    for k, v in c["env"].items():
        if hasattr(cfg.env, k):
            setattr(cfg.env, k, tuple(v) if isinstance(v, list) else v)
    for k, v in c["reward"].items():
        if hasattr(cfg.reward, k):
            setattr(cfg.reward, k, v)
    return cfg, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--memo", default=os.environ.get("SM_RL_SPINS_MEMO"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg, c = load_cfg(a.run_dir)
    eng = WolframEngine(cfg.engine).start()
    if a.memo:
        print(f"memo loaded: {eng.load_memo(a.memo)} entries")
    env = SMModelEnv(eng, cfg, cache=SpectrumCache(None, max_entries=200_000))
    tok = Tokenizer(eng, env.ladder, cfg.env.n_u1)
    pol = ActorCritic(GROUP_FEAT, tok.parton_feat, env.num_actions, max_partons=cfg.env.max_partons + 4)
    ck = torch.load(os.path.join(a.run_dir, "checkpoints", a.ckpt), map_location="cpu", weights_only=False)
    pol.load_state_dict(ck["policy"]); pol.eval()
    pool = [s.gm for s in env.ladder.specs if s.lie_rank <= 4]
    kinds = [d.kind for d in env.aspace.actions]
    kind_names = sorted(set(kinds)); kind_idx = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in kind_names}
    use_cursor = cfg.env.use_cursor
    print(f"run={a.run_dir} ckpt={a.ckpt} iters_done={ck.get('iters_done')} actions={env.num_actions} "
          f"charge_mode={cfg.env.charge_action_mode} cursor={use_cursor} max_steps={cfg.env.max_steps}")

    def run_episode(start, greedy):
        obs, _ = env.reset(start_group=start)
        H_total, H_kind, H_within, n = 0.0, 0.0, 0.0, 0
        kind_counts = collections.Counter(); traj_best = (-1, None)
        for _ in range(cfg.env.max_steps):
            f = tok.featurize(obs, use_cursor); b = to_torch(tok.collate([f]))
            with torch.no_grad():
                logits, _ = pol(b)
            p = torch.softmax(logits[0], 0).numpy()
            pk = np.array([p[kind_idx[k]].sum() for k in kind_names])
            H_total += float(-(p[p > 0] * np.log(p[p > 0])).sum())
            H_kind += float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
            if "charge_set" in kind_idx and pk[kind_names.index("charge_set")] > 1e-6:
                q = p[kind_idx["charge_set"]]; q = q / q.sum()
                H_within += float(-(q[q > 0] * np.log(q[q > 0])).sum())
            n += 1
            act = int(np.argmax(p)) if greedy else int(np.random.choice(len(p), p=p))
            kind_counts[kinds[act]] += 1
            obs, r, term, trunc, info = env.step(act)
            if info["score"] > traj_best[0]:
                traj_best = (info["score"], env.model.copy())
            if term or trunc:
                break
        m = env.model
        return dict(group=m.group, n=m.n, spins=tuple(p.spin for p in m.partons),
                    reps=tuple(p.rep_index for p in m.partons), score=info["score"], exact=info.get("exact", False),
                    H=H_total / n, Hk=H_kind / n, Hw=H_within / n, kinds=kind_counts, best=traj_best)

    for label, greedy, starts in [("GREEDY (1 per start group)", True, pool),
                                  (f"SAMPLED ({a.episodes} eps, random starts)", False,
                                   [pool[i % len(pool)] for i in range(a.episodes)])]:
        eps = [run_episode(s, greedy) for s in starts]
        print(f"\n===== {label}")
        sc = np.array([e["score"] for e in eps])
        print(f"final score: mean={sc.mean():.3f} median={np.median(sc):.3f} max={sc.max():.3f} "
              f"exact={sum(e['exact'] for e in eps)} | entropy/step: total={np.mean([e['H'] for e in eps]):.2f} "
              f"over-kinds={np.mean([e['Hk'] for e in eps]):.2f} within-charge_set={np.mean([e['Hw'] for e in eps]):.2f}")
        g = collections.Counter(e["group"] for e in eps)
        print("final group:", ", ".join(f"{k}:{v}({np.mean([e['score'] for e in eps if e['group']==k]):.2f})" for k, v in g.most_common()))
        print("n_partons:", dict(sorted(collections.Counter(e["n"] for e in eps).items())),
              "| all-fermion:", sum(all(s == 2 for s in e["spins"]) and e["n"] > 0 for e in eps),
              "| reps:", dict(collections.Counter(e["reps"] for e in eps).most_common(4)))
        kc = sum((e["kinds"] for e in eps), collections.Counter()); tot = sum(kc.values())
        print("action kinds:", ", ".join(f"{k}:{v/tot:.2f}" for k, v in kc.most_common()))
        b = max(eps, key=lambda e: e["best"][0])["best"]
        print(f"best model seen: {b[0]:.3f} {b[1].describe()}")
        su3 = [e for e in eps if e["group"] == "SU3"]
        if su3:
            print(f"SU3 episodes: {len(su3)} mean={np.mean([e['score'] for e in su3]):.3f} "
                  f"n_partons={dict(collections.Counter(e['n'] for e in su3))}")
    print(f"\nengine: {eng.memo_stats()}")
    eng.stop()


if __name__ == "__main__":
    main()
