from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .blocks import RMSNorm


def rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RefXAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.nq = RMSNorm(d_model)
        self.nkv = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True, dropout=dropout
        )
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.gmax = 0.35

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.nq(x)
        kv = self.nkv(ref.float())

        with torch.autocast(device_type=q.device.type, enabled=False):
            a, _ = self.attn(
                q.float(), kv, kv, key_padding_mask=key_padding_mask, need_weights=False
            )

        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        rms_x = rms(x.float())
        rms_a = rms(a)
        scale = rms_x / rms_a
        a = (a * scale).to(x.dtype)

        gate_eff = (self.gmax * torch.tanh(self.gate)).to(x.dtype)
        return x + gate_eff * a


class RefXAttn(nn.Module):
    def __init__(
        self, d_model: int, heads: int = 2, layers: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RefXAttnBlock(d_model, heads, dropout) for _ in range(layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, ref, key_padding_mask)
        return x


class TextXAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.nq = RMSNorm(d_model)
        self.nkv = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.nq(x)
        kv = self.nkv(context)
        with torch.autocast(device_type=q.device.type, enabled=False):
            out, _ = self.attn(
                q.float(),
                kv.float(),
                kv.float(),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype)
        return x + torch.tanh(self.gate) * out
