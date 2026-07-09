"""A short, honest PPO trial: NO curriculum, NO knowledge of the SM.

Every episode starts from the smallest group (SU(2)) with an empty model, so the
agent must discover the gauge group AND the matter content on its own. This is a
"does the loop behave" run, not a solve -- rediscovering (u,d,s) in a handful of
episodes from a random init is not expected.

Usage: ~/miniforge3/envs/jax/bin/python scripts/trial_ppo.py [iterations] [rollout_len]
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.config import Config
from sm_rl.physics import WolframEngine
from sm_rl.env import SMModelEnv
from sm_rl.models import Tokenizer, ActorCritic, GROUP_FEAT
from sm_rl.algos import PPO, PPOConfig


def describe(model):
    if model is None:
        return "None"
    ps = [f"r{p.rep_index}/s{p.spin}/q{[round(c,2) for c in p.charges]}" for p in model.partons]
    return f"{model.group}[" + ", ".join(ps) + "]"


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rollout_len = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = Config()
    eng = WolframEngine(cfg.engine).start()
    try:
        env = SMModelEnv(eng, cfg)
        tok = Tokenizer(eng, env.ladder, cfg.env.n_u1)
        policy = ActorCritic(GROUP_FEAT, tok.parton_feat, env.num_actions,
                             max_partons=cfg.env.max_partons + 4)
        # start_group=None  ->  every episode begins at the ladder default, SU(2)
        ppo = PPO(env, policy, tok,
                  PPOConfig(rollout_len=rollout_len, minibatch_size=min(32, rollout_len),
                            start_group=None))
        print(f"start group = {env.ladder.default.gm} (empty) | target = {len(env.target)} hadrons "
              f"| {iterations} iters x {rollout_len} steps\n")

        hist = ppo.train(iterations)

        print(f"{'iter':>4} {'steps':>6} {'eps':>4} {'mean_r':>8} {'ep_ret':>9} "
              f"{'best_sc':>8} {'ent':>6} {'pg':>9} {'vf':>9} {'exact':>6}")
        gbest_s, gbest_m = -1e9, None
        for h in hist:
            if h["best_score"] > gbest_s:
                gbest_s, gbest_m = h["best_score"], h["best_model"]
            print(f"{h['iter']:>4} {rollout_len:>6} {h['n_episodes']:>4} {h['mean_reward']:>8.2f} "
                  f"{h['ep_return']:>9.2f} {h['best_score']:>8.1f} {h['entropy']:>6.3f} "
                  f"{h['pg_loss']:>9.4f} {h['v_loss']:>9.1f} {h['exact_hits']:>6}")

        print(f"\nbest model found (score {gbest_s:+.1f}, 0 = exact SM):")
        print(f"   {describe(gbest_m)}")
        print("\nTRIAL: loop healthy" if all(np.isfinite([h['pg_loss'], h['v_loss'], h['entropy']]).all()
                                             for h in hist) else "TRIAL: numerical issue!")
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
