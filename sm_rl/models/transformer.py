"""Transformer actor-critic over [CLS, GROUP, P_1..P_n].

  * Group token and parton tokens are projected to d_model; a learned segment
    embedding tags CLS / GROUP / PARTON.
  * Partons get sinusoidal positional encodings (ordering is bookkeeping so the
    policy can track "the last-added" parton -- see addendum).
  * The CLS output drives a flat, masked action head and a value head.

The network is deliberately small (few labeled rewards). It is decoupled from
the RL algorithm: PPO, GFlowNet, or anything else consumes `logits, value`.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


def _sinusoidal(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ActorCritic(nn.Module):
    def __init__(self, group_feat: int, parton_feat: int, n_actions: int,
                 d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 ffn: int = 512, max_partons: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.group_proj = nn.Linear(group_feat, d_model)
        self.parton_proj = nn.Linear(parton_feat, d_model)
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.seg = nn.Embedding(3, d_model)              # 0=CLS, 1=GROUP, 2=PARTON
        self.register_buffer("pe", _sinusoidal(max_partons, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ffn, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                         nn.Linear(d_model, n_actions))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 1))

    def forward(self, batch: dict):
        """batch: torch tensors group_feat (B,Gf), parton_feat (B,N,Pf),
        pad_mask (B,N) bool, action_mask (B,A) bool. Returns logits (B,A), value (B,)."""
        gf, pf = batch["group_feat"], batch["parton_feat"]
        pad, amask = batch["pad_mask"], batch["action_mask"]
        B, N, _ = pf.shape

        cls = self.cls.expand(B, 1, self.d_model) + self.seg(torch.zeros(B, 1, dtype=torch.long, device=gf.device))
        grp = self.group_proj(gf).unsqueeze(1) + self.seg(torch.ones(B, 1, dtype=torch.long, device=gf.device))
        par = self.parton_proj(pf) + self.pe[:N].unsqueeze(0) + self.seg.weight[2]
        tokens = torch.cat([cls, grp, par], dim=1)       # (B, 2+N, d)

        # key padding mask: CLS & GROUP always visible; partons per pad.
        head = torch.zeros(B, 2, dtype=torch.bool, device=gf.device)
        key_pad = torch.cat([head, pad], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        h_cls = h[:, 0]

        logits = self.policy_head(h_cls)
        logits = torch.where(amask, logits, torch.full_like(logits, NEG_INF))
        value = self.value_head(h_cls).squeeze(-1)
        return logits, value

    # ---- RL helpers -------------------------------------------------------

    def act(self, batch: dict):
        logits, value = self.forward(batch)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, batch: dict, actions: torch.Tensor):
        logits, value = self.forward(batch)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def to_torch(collated: dict, device="cpu") -> dict:
    return {
        "group_feat": torch.as_tensor(collated["group_feat"], dtype=torch.float32, device=device),
        "parton_feat": torch.as_tensor(collated["parton_feat"], dtype=torch.float32, device=device),
        "pad_mask": torch.as_tensor(collated["pad_mask"], dtype=torch.bool, device=device),
        "action_mask": torch.as_tensor(collated["action_mask"], dtype=torch.bool, device=device),
    }
