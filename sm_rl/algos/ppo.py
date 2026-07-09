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
    minibatch_size: int = 32
    rollout_len: int = 64
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    start_group: str | None = "SU3"


class PPO(Algorithm):
    def __init__(self, env, policy, tokenizer, cfg: PPOConfig | None = None, device="cpu"):
        super().__init__(env, policy, tokenizer, device)
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
        self.iters_done = 0
        self.best_score = -float("inf")
        self.best_model = None

    def train(self, iterations: int, logger=None, ckpt_every: int = 0):
        history = []
        for _ in range(iterations):
            it = self.iters_done
            roll = collect_rollout(self.env, self.policy, self.tok,
                                   self.env.cfg.env.use_cursor, self.cfg.rollout_len,
                                   start_group=self.cfg.start_group, device=self.device)
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
            if logger is not None:
                logger.log(
                    f"iter {it:>4} mean_r={stats['mean_reward']:+.2f} "
                    f"ep_ret={stats['ep_return']:+.2f} best={self.best_score:+.1f} "
                    f"ent={stats['entropy']:.3f} pg={stats['pg_loss']:+.4f} "
                    f"vf={stats['v_loss']:.1f} eps={stats['n_episodes']} "
                    f"exact={stats['exact_hits']} hit_rate={cstats.get('cache_hit_rate', '-')}")
                logger.log_metrics({
                    "iter": it, "mean_reward": stats["mean_reward"], "ep_return": stats["ep_return"],
                    "best_score": self.best_score, "entropy": stats["entropy"],
                    "pg_loss": stats["pg_loss"], "v_loss": stats["v_loss"],
                    "n_episodes": stats["n_episodes"], "exact_hits": stats["exact_hits"],
                    **cstats})

            if logger is not None and ckpt_every and self.iters_done % ckpt_every == 0:
                self.save(os.path.join(logger.ckpt_dir, f"ckpt_{self.iters_done:06d}.pt"))
                self.save(os.path.join(logger.ckpt_dir, "latest.pt"))
                if hasattr(self.env.cache, "flush"):
                    self.env.cache.flush()
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

    def _update(self, roll: dict) -> dict:
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
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                last = {"pg_loss": float(pg_loss.item()), "v_loss": float(v_loss.item()),
                        "entropy": float(ent.item())}
        return last
