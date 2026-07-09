"""RL algorithm interface + a shared rollout collector.

The env (sm_env) and the policy network (models.transformer) are both algorithm
-agnostic. An Algorithm ties them together and defines the learning rule. To
swap PPO for e.g. a GFlowNet, subclass Algorithm and reuse `collect_rollout`
(and, if desired, the same ActorCritic encoder). Nothing else changes.
"""
from __future__ import annotations

import abc

import numpy as np
import torch

from ..models.transformer import to_torch


def collect_rollout(env, policy, tok, use_cursor: bool, n_steps: int,
                    start_group: str | None = None, device: str = "cpu") -> dict:
    """Run `policy` in `env` for n_steps, auto-resetting on episode end.
    Returns per-step feats/actions/logps/values/rewards/dones + bootstrap value.
    """
    feats, actions, logps, values, rewards, dones = [], [], [], [], [], []
    ep_returns, cur_return = [], 0.0
    exact_hits = 0
    best_score, best_model = -float("inf"), None

    if env.model is None:
        obs, _ = env.reset(start_group=start_group)
    else:
        obs = env._obs()

    for _ in range(n_steps):
        f = tok.featurize(obs, use_cursor)
        batch = to_torch(tok.collate([f]), device)
        with torch.no_grad():
            a, lp, v = policy.act(batch)
        a = int(a.item())
        feats.append(f); actions.append(a)
        logps.append(float(lp.item())); values.append(float(v.item()))

        obs, r, term, trunc, info = env.step(a)
        rewards.append(r); cur_return += r
        exact_hits += int(term and info.get("exact", False))
        if info["score"] > best_score:
            best_score, best_model = info["score"], env.model.copy()
        done = term or trunc
        dones.append(done)
        if done:
            ep_returns.append(cur_return); cur_return = 0.0
            obs, _ = env.reset(start_group=start_group)

    f = tok.featurize(obs, use_cursor)
    batch = to_torch(tok.collate([f]), device)
    with torch.no_grad():
        _, last_v = policy.forward(batch)

    return dict(feats=feats, actions=actions, logps=logps, values=values,
                rewards=rewards, dones=dones, last_value=float(last_v.item()),
                ep_returns=ep_returns, exact_hits=exact_hits,
                best_score=best_score, best_model=best_model)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


class Algorithm(abc.ABC):
    def __init__(self, env, policy, tokenizer, device: str = "cpu"):
        self.env = env
        self.policy = policy
        self.tok = tokenizer
        self.device = device

    @abc.abstractmethod
    def train(self, iterations: int): ...
