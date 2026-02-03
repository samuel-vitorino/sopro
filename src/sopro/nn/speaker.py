from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentiveStatsPool, DepthwiseConv1d


class Token2SV(nn.Module):
    def __init__(
        self, Q: int, V: int, d: int = 192, out_dim: int = 192, dropout: float = 0.05
    ):
        super().__init__()
        self.Q, self.V = int(Q), int(V)
        self.emb = nn.Embedding(self.Q * self.V, d)

        initial_weights = torch.linspace(1.0, 0.1, self.Q)
        self.cb_weights = nn.Parameter(initial_weights)

        self.enc = nn.Sequential(
            DepthwiseConv1d(d, 7, causal=False),
            nn.GELU(),
            nn.Dropout(dropout),
            DepthwiseConv1d(d, 7, causal=False),
            nn.GELU(),
        )
        self.pool = AttentiveStatsPool(d)
        self.proj = nn.Linear(2 * d, out_dim)

    def _get_mixed_embedding(self, embed_btqd: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.cb_weights, dim=0).view(1, 1, self.Q, 1)
        return (embed_btqd * w).sum(dim=2)

    def forward(
        self, tokens_btq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, Q = tokens_btq.shape
        device = tokens_btq.device

        if lengths is not None:
            valid = torch.arange(T, device=device)[None, :] < lengths[:, None]
        else:
            valid = torch.ones(B, T, device=device, dtype=torch.bool)

        q_idx = torch.arange(Q, device=device, dtype=torch.long).view(1, 1, Q)
        idx = q_idx * self.V + tokens_btq.long()
        raw_emb = self.emb(idx)
        raw_emb = raw_emb * valid[:, :, None, None].to(raw_emb.dtype)

        x = self._get_mixed_embedding(raw_emb)
        x = x * valid[:, :, None].to(x.dtype)

        h = self.enc(x)
        h = h * valid[:, :, None].to(h.dtype)

        pooled = self.pool(h, lengths=lengths)
        e = self.proj(pooled)
        return F.normalize(e, dim=-1, eps=1e-6)


class SpeakerFiLM(nn.Module):
    def __init__(self, d_model: int, sv_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sv_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, base_bt_d: torch.Tensor, spk_b_d: torch.Tensor, strength: float = 1.0
    ) -> torch.Tensor:
        B, T, D = base_bt_d.shape
        film = self.mlp(spk_b_d)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(B, T, D)
        beta = beta.unsqueeze(1).expand(B, T, D)
        x = self.norm(base_bt_d)
        return x * (1 + strength * torch.tanh(gamma)) + strength * torch.tanh(beta)
