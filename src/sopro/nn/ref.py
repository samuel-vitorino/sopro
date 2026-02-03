from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm


def _rms_per_token(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


class RefXAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 2, gmax: float = 0.35):
        super().__init__()
        assert d_model % heads == 0

        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = self.d_model // self.heads
        self.gmax = float(gmax)

        self.nq = RMSNorm(self.d_model)
        self.nkv = RMSNorm(self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.gate = nn.Parameter(torch.tensor(0.0))

    def _to_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, D = t.shape
        return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, Hd = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def build_kv_cache(
        self,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kv = self.nkv(context)
        k = self._to_heads(self.k_proj(kv))
        v = self._to_heads(self.v_proj(kv))
        return {"k": k, "v": v, "key_padding_mask": key_padding_mask}

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        q = self.nq(x)
        q = self._to_heads(self.q_proj(q))

        if kv_cache is None:
            if context is None:
                raise ValueError("context must be provided when kv_cache is None")
            kv_cache = self.build_kv_cache(context, key_padding_mask=key_padding_mask)

        k = kv_cache["k"]
        v = kv_cache["v"]
        kpm = kv_cache.get("key_padding_mask", None)

        attn_bias = None
        if kpm is not None:
            kpm = kpm.to(torch.bool)
            B = q.size(0)
            S = k.size(-2)
            attn_bias = torch.zeros((B, 1, 1, S), device=q.device, dtype=torch.float32)
            attn_bias = attn_bias.masked_fill(kpm[:, None, None, :], float("-inf"))

            bad = kpm.all(dim=1)
            if bad.any():
                attn_bias = attn_bias.clone()
                attn_bias[bad, :, :, 0] = 0.0

        with torch.autocast(device_type=x.device.type, enabled=False):
            a = F.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False,
            )

        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = self._from_heads(a)

        scale = (_rms_per_token(x) / _rms_per_token(a)).clamp(0.0, 10.0)
        a = (a * scale).to(x.dtype)

        a = self.out_proj(a)

        gate_eff = (self.gmax * torch.tanh(self.gate)).to(x.dtype)
        y = x + gate_eff * a
        return (y, kv_cache) if use_cache else y


class RefXAttnStack(nn.Module):
    def __init__(
        self, d_model: int, heads: int = 2, layers: int = 3, gmax: float = 0.35
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RefXAttnBlock(d_model, heads=heads, gmax=gmax) for _ in range(int(layers))]
        )

    def build_kv_caches(
        self,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            blk.build_kv_cache(context, key_padding_mask=key_padding_mask)
            for blk in self.blocks
        ]

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        if use_cache:
            if kv_caches is None:
                if context is None:
                    raise ValueError("context must be provided when kv_caches is None")
                kv_caches = self.build_kv_caches(
                    context, key_padding_mask=key_padding_mask
                )

            assert kv_caches is not None and len(kv_caches) == len(self.blocks)
            new_caches: List[Dict[str, torch.Tensor]] = []
            h = x
            for blk, cache in zip(self.blocks, kv_caches):
                h, cache2 = blk(h, kv_cache=cache, use_cache=True)
                new_caches.append(cache2)
            return h, new_caches

        if context is None:
            raise ValueError("context must be provided when use_cache=False")
        h = x
        for blk in self.blocks:
            h = blk(h, context=context, key_padding_mask=key_padding_mask)
        return h
