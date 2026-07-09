"""Wire engine + env + tokenizer + transformer policy + PPO and train.

Usage:
  ~/miniforge3/envs/jax/bin/python scripts/train_ppo.py [iterations] [rollout_len]

With no args it runs a short smoke (verifies the loop: shapes, gradients, no
NaNs, entropy sane). Longer runs attempt to actually rediscover (u,d,s).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # before numpy/scipy/torch
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.config import Config
from sm_rl.physics import WolframEngine
from sm_rl.env import SMModelEnv
from sm_rl.models import Tokenizer, ActorCritic, GROUP_FEAT
from sm_rl.algos import PPO, PPOConfig


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rollout_len = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = Config()
    eng = WolframEngine(cfg.engine).start()
    try:
        env = SMModelEnv(eng, cfg)
        tok = Tokenizer(eng, env.ladder, cfg.env.n_u1)
        policy = ActorCritic(
            group_feat=GROUP_FEAT, parton_feat=tok.parton_feat,
            n_actions=env.num_actions, max_partons=cfg.env.max_partons + 4)
        nparams = sum(p.numel() for p in policy.parameters())
        print(f"policy params: {nparams:,} | actions: {env.num_actions} | target: {len(env.target)} states")

        ppo = PPO(env, policy, tok, PPOConfig(rollout_len=rollout_len,
                                              minibatch_size=min(32, rollout_len)))
        hist = ppo.train(iterations)
        print(f"\n{'iter':>4} {'mean_r':>8} {'ep_ret':>8} {'pg':>9} {'vf':>8} {'ent':>6} {'exact':>6}")
        for h in hist:
            print(f"{h['iter']:>4} {h['mean_reward']:>8.2f} {h['ep_return']:>8.2f} "
                  f"{h['pg_loss']:>9.4f} {h['v_loss']:>8.2f} {h['entropy']:>6.3f} {h['exact_hits']:>6}")

        # correctness assertions for the smoke run
        for h in hist:
            assert np.isfinite(h["pg_loss"]) and np.isfinite(h["v_loss"]) and np.isfinite(h["entropy"])
        assert hist[-1]["entropy"] > 0.0
        print("\nTRAIN_PPO SMOKE: OK")
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
