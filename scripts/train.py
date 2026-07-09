"""Full PPO training runner: file logging + checkpointing + persistent cache.

Creates (or reuses) a run directory under runs/ containing log.txt, metrics.csv,
config.json, best_model.json, checkpoints/, and cache.pkl. Safe to interrupt and
resume.

Examples:
  # fresh run (no curriculum, starts from SU(2), no SM knowledge)
  uv run python scripts/train.py --iters 200 --rollout 64 --ckpt-every 10

  # resume the latest checkpoint of an existing run, 100 more iterations
  uv run python scripts/train.py --run-dir runs/ppo_20260709_101500 --resume latest --iters 100

Flags: --iters --rollout --start-group SU2 --metric count|hungarian
       --run-name --run-dir --resume PATH|'latest' --ckpt-every --cache PATH --seed
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sm_rl.config import Config
from sm_rl.physics import WolframEngine, SpectrumCache
from sm_rl.env import SMModelEnv
from sm_rl.models import Tokenizer, ActorCritic, GROUP_FEAT
from sm_rl.algos import PPO, PPOConfig
from sm_rl.utils import Logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--start-group", default=None, help="episode start group (default: SU2)")
    p.add_argument("--metric", default=None, choices=["count", "hungarian"],
                   help="reward metric; default 'hungarian' for new runs, "
                        "auto-reused from the run's config.json on resume")
    p.add_argument("--run-name", default="ppo")
    p.add_argument("--run-dir", default=None, help="reuse an existing run dir (for resume)")
    p.add_argument("--resume", default=None, help="checkpoint path, or 'latest'")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--cache", default=None, help="spectrum cache path (default: <run_dir>/cache.pkl)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = Config()
    # reward metric: explicit --metric > existing run's config.json (resume) > hungarian
    metric = a.metric
    if metric is None:
        prev = os.path.join(a.run_dir, "config.json") if a.run_dir else None
        if prev and os.path.exists(prev):
            import json
            try:
                metric = json.load(open(prev)).get("reward", {}).get("metric")
            except Exception:
                metric = None
        metric = metric or "hungarian"
    cfg.reward.metric = metric

    logger = Logger(run_dir=a.run_dir, name=a.run_name)
    cache_path = a.cache or os.path.join(logger.run_dir, "cache.pkl")
    logger.save_json("config.json", {
        "argv": vars(a), "cache_path": cache_path,
        "env": cfg.env.__dict__, "groups": cfg.groups.__dict__, "reward": cfg.reward.__dict__,
    })

    eng = WolframEngine(cfg.engine).start()
    try:
        cache = SpectrumCache(cache_path, flush_every=200)
        logger.log(f"spectrum cache: {cache_path}  (loaded {len(cache)} entries)")
        env = SMModelEnv(eng, cfg, cache=cache)
        tok = Tokenizer(eng, env.ladder, cfg.env.n_u1)
        policy = ActorCritic(GROUP_FEAT, tok.parton_feat, env.num_actions,
                             max_partons=cfg.env.max_partons + 4)
        ppo = PPO(env, policy, tok,
                  PPOConfig(rollout_len=a.rollout, minibatch_size=min(32, a.rollout),
                            start_group=a.start_group))

        if a.resume:
            path = a.resume
            if path == "latest":
                path = os.path.join(logger.ckpt_dir, "latest.pt")
            ppo.load(path)
            logger.log(f"resumed from {path} at iter {ppo.iters_done} (best={ppo.best_score:+.1f})")

        logger.log(f"target={len(env.target)} hadrons | actions={env.num_actions} | "
                   f"params={sum(p.numel() for p in policy.parameters()):,} | "
                   f"start_group={a.start_group or env.ladder.default.gm} | metric={metric}")
        ppo.train(a.iters, logger=logger, ckpt_every=a.ckpt_every)

        ppo.save(os.path.join(logger.ckpt_dir, "final.pt"))
        cache.flush()
        logger.log(f"done. best score {ppo.best_score:+.1f} "
                   f"({ppo.best_model.describe() if ppo.best_model else 'None'})")
        logger.log(f"cache: {cache.stats()}")
    finally:
        eng.stop()
        logger.close()


if __name__ == "__main__":
    main()
