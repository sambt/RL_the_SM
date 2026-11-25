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
    p.add_argument("--rollout", type=int, default=1024, help="steps collected per PPO update")
    p.add_argument("--minibatch", type=int, default=256, help="PPO minibatch size")
    p.add_argument("--start-group", default=None,
                   help="episode start group for --start-mode fixed (default: SU2)")
    p.add_argument("--start-mode", default="fixed",
                   choices=["fixed", "random", "curriculum"],
                   help="'fixed': always --start-group. 'random': draw a fresh "
                        "group per episode from --start-pool. 'curriculum': start "
                        "at --curriculum-base, anneal to random over "
                        "--curriculum-anneal-iters.")
    p.add_argument("--start-pool", default=None,
                   help="comma-separated group symbols the random/curriculum "
                        "schedules draw from (e.g. 'SU2,SU3,SO5,G2'), or 'all' for "
                        "the whole ladder. Default: curated low-rank (Lie rank<=4).")
    p.add_argument("--curriculum-base", default="SU3",
                   help="easy anchor group for --start-mode curriculum")
    p.add_argument("--curriculum-anneal-iters", type=int, default=2000,
                   help="iters over which curriculum p(base) ramps 1->0")
    p.add_argument("--terminal-mode", default=None,
                   choices=["exact", "graded", "shaped"],
                   help="how the STOP bonus is paid (default 'exact'; 'graded'/"
                        "'shaped' turn the exact-match cliff into a slope)")
    p.add_argument("--terminal-bonus", type=float, default=None,
                   help="STOP bonus scale (default 10.0; use ~2-3 with graded/shaped)")
    p.add_argument("--no-stop", dest="allow_stop", action="store_false",
                   help="mask the STOP action entirely: episodes end only at "
                        "max_steps or on an exact SM match (fixed-horizon setup). "
                        "The terminal bonus is then paid only on an exact match.")
    p.set_defaults(allow_stop=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="episode step cap (EnvConfig.max_steps; default 128). "
                        "Lower (~48) => shorter episodes => more episodes/update.")
    p.add_argument("--charge-actions", default=None, choices=["step", "set"],
                   help="'step' (legacy +/-1 increments) or 'set' (one action per "
                        "lattice value per charge: any value is one move away). "
                        "The lattice range/steps are unchanged either way.")
    p.add_argument("--cursor", dest="use_cursor", action="store_true", default=None,
                   help="add CURSOR_PREV/NEXT so edits can target ANY parton "
                        "(default: edits hit the last parton only, so changing an "
                        "earlier one means removing everything after it)")
    p.add_argument("--novelty-coef", type=float, default=None,
                   help="novelty bonus scale beta (default 0=off); keyed on "
                        "physics (canonical_key). ~0.01 with first_visit mode.")
    p.add_argument("--novelty-mode", default=None, choices=["count", "first_visit"],
                   help="'count' (beta/sqrt(n) every visit, legacy) or 'first_visit' "
                        "(beta once on first visit, 0 after; self-fading, preferred)")
    p.add_argument("--novelty-anneal-iters", type=int, default=None,
                   help="linear horizon to anneal the novelty bonus -> 0 (0=off)")
    p.add_argument("--epochs", type=int, default=None,
                   help="PPO update epochs per rollout (default 4; lower=gentler)")
    p.add_argument("--gamma", type=float, default=None, help="discount (default 0.99)")
    p.add_argument("--ent-coef", type=float, default=None,
                   help="entropy coef at iter 0 (default 0.03)")
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="entropy coef floor after anneal (default 0.005; raise to "
                        "~0.02 to keep exploring instead of collapsing)")
    p.add_argument("--ent-anneal-iters", type=int, default=None,
                   help="linear entropy-anneal horizon (default 5000; 0=fixed)")
    p.add_argument("--metric", default=None, choices=["count", "hungarian", "f1"],
                   help="reward metric; default 'f1' for new runs, "
                        "auto-reused from the run's config.json on resume")
    p.add_argument("--normalization", default="physical", choices=["gcd", "none", "physical"],
                   help="charge normalization before matching (default 'physical': "
                        "undoes the x3 scaled-integer convention so the charge quantum "
                        "is 1 and near-misses earn partial credit; 'none' leaves the "
                        "quantum at 3, wider than the match radius -> no gradient)")
    p.add_argument("--run-name", default="ppo")
    p.add_argument("--run-dir", default=None, help="reuse an existing run dir (for resume)")
    p.add_argument("--resume", default=None, help="checkpoint path, or 'latest'")
    p.add_argument("--init-from", default=None,
                   help="warm-start: load ONLY policy weights from this checkpoint "
                        "into a NEW run (iters/optimizer/best reset, schedules "
                        "restart from 0). For seeding from a prior run's policy. "
                        "Mutually exclusive with --resume.")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--cache", default=None,
                   help="charge-keyed spectrum cache pickle to load/flush. Precedence: "
                        "--cache > $SM_RL_CACHE > none (in-memory only, bounded by "
                        "--cache-max-entries). Optional now that the GetSpins memo is "
                        "persisted (--spins-memo): a spectrum miss costs only Python. "
                        "Loading the ~5 GB shared file costs ~20 GB RAM; avoid.")
    p.add_argument("--cache-max-entries", type=int, default=200_000,
                   help="cap on in-memory spectrum-cache entries (cleared when hit)")
    p.add_argument("--spins-memo", default=None,
                   help="persisted engine memo (charge-free GetSpins results; KB-MB). "
                        "Precedence: --spins-memo > $SM_RL_SPINS_MEMO > "
                        "<run_dir>/spins_memo.pkl. Loaded at start, merged+saved at "
                        "every checkpoint and at exit. Share one file across runs.")
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
        metric = metric or "f1"
    cfg.reward.metric = metric
    cfg.reward.normalization = a.normalization
    if a.terminal_mode is not None:
        cfg.env.terminal_mode = a.terminal_mode
    if a.terminal_bonus is not None:
        cfg.env.terminal_bonus = a.terminal_bonus
    cfg.env.allow_stop = a.allow_stop
    if a.max_steps is not None:
        cfg.env.max_steps = a.max_steps
    if a.charge_actions is not None:
        cfg.env.charge_action_mode = a.charge_actions
    if a.use_cursor:
        cfg.env.use_cursor = True
    if a.novelty_coef is not None:
        cfg.env.novelty_coef = a.novelty_coef
    if a.novelty_mode is not None:
        cfg.env.novelty_mode = a.novelty_mode

    logger = Logger(run_dir=a.run_dir, name=a.run_name)
    cache_path = a.cache or os.environ.get("SM_RL_CACHE") or None
    memo_path = (a.spins_memo or os.environ.get("SM_RL_SPINS_MEMO")
                 or os.path.join(logger.run_dir, "spins_memo.pkl"))
    logger.save_json("config.json", {
        "argv": vars(a), "cache_path": cache_path, "spins_memo": memo_path,
        "env": cfg.env.__dict__, "groups": cfg.groups.__dict__, "reward": cfg.reward.__dict__,
    })

    eng = WolframEngine(cfg.engine).start()
    try:
        n_memo = eng.load_memo(memo_path)
        logger.log(f"spins memo: {memo_path}  (loaded {n_memo} entries)")
        cache = SpectrumCache(cache_path, flush_every=2000, max_entries=a.cache_max_entries)
        logger.log(f"spectrum cache: {cache_path or 'in-memory'}  (loaded {len(cache)} entries, "
                   f"max {a.cache_max_entries})")
        env = SMModelEnv(eng, cfg, cache=cache)
        tok = Tokenizer(eng, env.ladder, cfg.env.n_u1)
        policy = ActorCritic(GROUP_FEAT, tok.parton_feat, env.num_actions,
                             max_partons=cfg.env.max_partons + 4)
        if a.start_pool is None:
            start_pool = None
        elif a.start_pool.lower() == "all":
            start_pool = tuple(s.gm for s in env.ladder.specs)
        else:
            start_pool = tuple(g.strip() for g in a.start_pool.split(",") if g.strip())
        ppo_cfg = PPOConfig(rollout_len=a.rollout, minibatch_size=min(a.minibatch, a.rollout),
                            start_group=a.start_group, start_mode=a.start_mode,
                            start_pool=start_pool, curriculum_base=a.curriculum_base,
                            curriculum_anneal_iters=a.curriculum_anneal_iters)
        if a.epochs is not None:
            ppo_cfg.epochs = a.epochs
        if a.gamma is not None:
            ppo_cfg.gamma = a.gamma
        if a.ent_coef is not None:
            ppo_cfg.ent_coef = a.ent_coef
        if a.ent_coef_final is not None:
            ppo_cfg.ent_coef_final = a.ent_coef_final
        if a.ent_anneal_iters is not None:
            ppo_cfg.ent_anneal_iters = a.ent_anneal_iters
        if a.novelty_anneal_iters is not None:
            ppo_cfg.novelty_anneal_iters = a.novelty_anneal_iters
        ppo = PPO(env, policy, tok, ppo_cfg)

        if a.resume and a.init_from:
            raise SystemExit("--resume and --init-from are mutually exclusive")
        if a.resume:
            path = a.resume
            if path == "latest":
                path = os.path.join(logger.ckpt_dir, "latest.pt")
            ppo.load(path)
            logger.log(f"resumed from {path} at iter {ppo.iters_done} (best={ppo.best_score:+.1f})")
        elif a.init_from:
            ppo.load_weights(a.init_from)
            logger.log(f"warm-started policy weights from {a.init_from} "
                       f"(iters/optimizer/best reset; schedules restart at 0)")

        if a.start_mode == "fixed":
            start_desc = f"fixed:{a.start_group or env.ladder.default.gm}"
        elif a.start_mode == "random":
            start_desc = f"random{list(ppo._start_pool)}"
        else:
            start_desc = (f"curriculum:{a.curriculum_base}->random"
                          f"(anneal {a.curriculum_anneal_iters}){list(ppo._start_pool)}")
        logger.log(f"target={len(env.target)} hadrons | actions={env.num_actions} | "
                   f"params={sum(p.numel() for p in policy.parameters()):,} | "
                   f"start={start_desc} | metric={metric} | "
                   f"terminal={cfg.env.terminal_mode}({cfg.env.terminal_bonus}) | "
                   f"stop={'on' if cfg.env.allow_stop else 'OFF (end=max_steps|exact)'} | "
                   f"max_steps={cfg.env.max_steps} | "
                   f"charge_actions={cfg.env.charge_action_mode} cursor={'on' if cfg.env.use_cursor else 'off'} | "
                   f"novelty={cfg.env.novelty_coef}({cfg.env.novelty_mode}"
                   f"{f',anneal{ppo_cfg.novelty_anneal_iters}' if ppo_cfg.novelty_anneal_iters else ''}) | "
                   f"epochs={ppo_cfg.epochs} ent={ppo_cfg.ent_coef}->{ppo_cfg.ent_coef_final}")
        ppo.train(a.iters, logger=logger, ckpt_every=a.ckpt_every, memo_path=memo_path)

        ppo.save(os.path.join(logger.ckpt_dir, "final.pt"))
        cache.flush()
        logger.log(f"done. best score {ppo.best_score:+.1f} "
                   f"({ppo.best_model.describe() if ppo.best_model else 'None'})")
        logger.log(f"cache: {cache.stats()} | memo: {eng.memo_stats()}")
    finally:
        try:
            n = eng.save_memo(memo_path)
            logger.log(f"spins memo saved: {memo_path} ({n} entries)")
        except Exception as e:
            logger.log(f"spins memo save FAILED: {e}")
        eng.stop()
        logger.close()


if __name__ == "__main__":
    main()
