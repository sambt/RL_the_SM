"""PPO with invalid-action masking (the default learner).

Standard clipped-surrogate PPO + GAE. Action masking is handled inside the
policy (invalid-action logits set to -inf), so sampling and log-probs already
respect the mask. Includes checkpoint save/load (resume-safe) and optional
file logging; the point of the framework is that this file can be replaced
wholesale by another Algorithm without touching env, tokenizer, or network.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .base import Algorithm, collect_rollout, compute_gae
from ..models.transformer import to_torch


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    rollout_len: int = 1024
    ent_coef: float = 0.03          # entropy bonus at iter 0 (annealed to ent_coef_final)
    ent_coef_final: float = 0.005   # entropy coef reached after ent_anneal_iters
    ent_anneal_iters: int = 5000    # linear-anneal horizon (0 = fixed ent_coef)
    # Linear anneal horizon for the env's novelty bonus: the learner scales
    # env._novelty_scale by max(0, 1 - iters_done/novelty_anneal_iters), so the
    # intrinsic bonus fades to 0 and late training exploits. 0 = no anneal.
    novelty_anneal_iters: int = 0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- episode start group ------------------------------------------------
    # start_mode:
    #   "fixed"      -> every episode starts at `start_group` (legacy behaviour).
    #   "random"     -> every episode starts at a group drawn uniformly from
    #                   `start_pool` (full exploration of the group axis).
    #   "curriculum" -> start at `curriculum_base` with prob (1-frac), else a
    #                   uniform draw from `start_pool`, where frac ramps 0->1
    #                   linearly over `curriculum_anneal_iters`. So training
    #                   begins anchored on an easy group and anneals to random.
    start_group: str | None = "SU3"
    start_mode: str = "fixed"
    start_pool: tuple[str, ...] | None = None   # None -> curated low-rank default
    curriculum_base: str = "SU3"
    curriculum_anneal_iters: int = 2000


class PPO(Algorithm):
    def __init__(self, env, policy, tokenizer, cfg: PPOConfig | None = None, device="cpu"):
        super().__init__(env, policy, tokenizer, device)
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
        self.iters_done = 0
        self.best_score = -float("inf")
        self.best_model = None
        self._start_pool = self._resolve_start_pool()

    # ---- episode start-group scheduling -----------------------------------

    def _resolve_start_pool(self) -> list[str]:
        """Group symbols the random / curriculum schedules draw from. Explicit
        `cfg.start_pool` wins; otherwise a curated default of the low-rank groups
        (Lie rank <= 4) spanning every family -- the regime where 3-body singlets
        can actually form at d_max=3, so episodes aren't wasted starting deep in
        (say) E8 where nothing binds."""
        if self.cfg.start_pool:
            return list(self.cfg.start_pool)
        pool = [s.gm for s in self.env.ladder.specs if s.lie_rank <= 4]
        return pool or [self.env.ladder.default.gm]

    def _start_group_arg(self):
        """Resolve cfg.start_mode into the `start_group` argument collect_rollout
        expects: a str for 'fixed', else a per-episode sampler closure. The
        curriculum mix fraction is keyed on iters_done, so it is resume-safe and
        constant within a single rollout."""
        c = self.cfg
        if c.start_mode == "fixed":
            return c.start_group
        pool = self._start_pool
        if c.start_mode == "random":
            p_base = 0.0
        elif c.start_mode == "curriculum":
            frac = min(1.0, self.iters_done / max(1, c.curriculum_anneal_iters))
            p_base = 1.0 - frac            # prob of anchoring on the easy base
        else:
            raise ValueError(f"unknown start_mode {c.start_mode!r}")
        base = c.curriculum_base

        def sample():
            if p_base > 0.0 and np.random.rand() < p_base:
                return base
            return pool[np.random.randint(len(pool))]
        return sample

    def train(self, iterations: int, logger=None, ckpt_every: int = 0,
              memo_path: str | None = None):
        history = []
        for _ in range(iterations):
            it = self.iters_done
            # Anneal the env's novelty bonus (keyed on iters_done -> resume-safe).
            if self.cfg.novelty_anneal_iters and self.cfg.novelty_anneal_iters > 0:
                self.env._novelty_scale = max(
                    0.0, 1.0 - it / self.cfg.novelty_anneal_iters)
            roll = collect_rollout(self.env, self.policy, self.tok,
                                   self.env.cfg.env.use_cursor, self.cfg.rollout_len,
                                   start_group=self._start_group_arg(), device=self.device)
            stats = self._update(roll)
            stats.update(
                iter=it,
                mean_reward=float(np.mean(roll["rewards"])),
                ep_return=float(np.mean(roll["ep_returns"])) if roll["ep_returns"] else float("nan"),
                n_episodes=len(roll["ep_returns"]),
                exact_hits=roll["exact_hits"],
                best_score_iter=roll["best_score"],
            )
            # track global best
            if roll["best_model"] is not None and roll["best_score"] > self.best_score:
                self.best_score = roll["best_score"]
                self.best_model = roll["best_model"]
                if logger is not None:
                    logger.save_json("best_model.json", {
                        "score": self.best_score, "iter": it,
                        "describe": self.best_model.describe(),
                        "model": self.best_model.to_dict()})
            self.iters_done = it + 1

            cstats = self.env.cache.stats() if hasattr(self.env.cache, "stats") else {}
            if hasattr(self.env.engine, "memo_stats"):
                cstats.update(self.env.engine.memo_stats())
            if logger is not None:
                logger.log(
                    f"iter {it:>4} mean_r={stats['mean_reward']:+.2f} "
                    f"ep_ret={stats['ep_return']:+.2f} best={self.best_score:+.1f} "
                    f"ent={stats['entropy']:.3f} ec={stats['ent_coef']:.3f} "
                    f"pg={stats['pg_loss']:+.4f} "
                    f"vf={stats['v_loss']:.1f} eps={stats['n_episodes']} "
                    f"exact={stats['exact_hits']} hit_rate={cstats.get('cache_hit_rate', '-')} "
                    f"spins_calls={cstats.get('spins_calls', '-')}")
                logger.log_metrics({
                    "iter": it, "mean_reward": stats["mean_reward"], "ep_return": stats["ep_return"],
                    "best_score": self.best_score, "entropy": stats["entropy"],
                    "ent_coef": stats["ent_coef"],
                    "pg_loss": stats["pg_loss"], "v_loss": stats["v_loss"],
                    "n_episodes": stats["n_episodes"], "exact_hits": stats["exact_hits"],
                    **cstats})

            if logger is not None and ckpt_every and self.iters_done % ckpt_every == 0:
                self.save(os.path.join(logger.ckpt_dir, f"ckpt_{self.iters_done:06d}.pt"))
                self.save(os.path.join(logger.ckpt_dir, "latest.pt"))
                if hasattr(self.env.cache, "flush"):
                    self.env.cache.flush()
                if memo_path and hasattr(self.env.engine, "save_memo"):
                    self.env.engine.save_memo(memo_path)
            history.append(stats)
        if hasattr(self.env.cache, "flush"):
            self.env.cache.flush()
        return history

    # ---- checkpointing ----------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "opt": self.opt.state_dict(),
            "iters_done": self.iters_done,
            "best_score": self.best_score,
            "best_model": self.best_model.to_dict() if self.best_model is not None else None,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "ppo_cfg": asdict(self.cfg),
        }, path)

    def load_weights(self, path: str) -> dict:
        """Warm-start: load ONLY the policy weights from a checkpoint, leaving
        iters_done=0, a fresh optimizer, and best reset. Unlike `load` (which
        resumes an interrupted run), this seeds a *new* run from another run's
        policy -- e.g. start from a checkpoint that already discovered the SM
        structure, then re-inflate exploration (entropy/novelty schedules restart
        from iter 0) to search onward. Architecture must match."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        return ck

    def load(self, path: str) -> dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["opt"])
        self.iters_done = ck.get("iters_done", 0)
        self.best_score = ck.get("best_score", -float("inf"))
        bm = ck.get("best_model")
        if bm is not None:
            from ..env.state import Model
            self.best_model = Model.from_dict(bm)
        try:
            torch.set_rng_state(ck["torch_rng"])
            np.random.set_state(ck["numpy_rng"])
        except Exception:
            pass
        return ck

    # ---- update -----------------------------------------------------------

    def _current_ent_coef(self) -> float:
        """Entropy coefficient, linearly annealed from ent_coef to ent_coef_final
        over the first ent_anneal_iters (then held). Keyed on iters_done so it is
        resume-safe."""
        c = self.cfg
        if c.ent_anneal_iters <= 0:
            return c.ent_coef
        frac = min(1.0, self.iters_done / c.ent_anneal_iters)
        return c.ent_coef + frac * (c.ent_coef_final - c.ent_coef)

    def _update(self, roll: dict) -> dict:
        ent_coef = self._current_ent_coef()
        adv, returns = compute_gae(roll["rewards"], roll["values"], roll["dones"],
                                   roll["last_value"], self.cfg.gamma, self.cfg.gae_lambda)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        act_t = torch.as_tensor(np.array(roll["actions"]), dtype=torch.long, device=self.device)
        logp_old = torch.as_tensor(np.array(roll["logps"]), dtype=torch.float32, device=self.device)

        batch = to_torch(self.tok.collate(roll["feats"]), self.device)
        T = len(roll["actions"])

        last = {}
        for _ in range(self.cfg.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.cfg.minibatch_size):
                mb = idx[s:s + self.cfg.minibatch_size]
                sub = {k: v[mb] for k, v in batch.items()}
                new_logp, entropy, value = self.policy.evaluate(sub, act_t[mb])
                ratio = torch.exp(new_logp - logp_old[mb])
                a = adv_t[mb]
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * a
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = ((value - ret_t[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                last = {"pg_loss": float(pg_loss.item()), "v_loss": float(v_loss.item()),
                        "entropy": float(ent.item()), "ent_coef": ent_coef}
        return last
