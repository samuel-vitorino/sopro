from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sopro.config import SoproTTSConfig
from sopro.nn.embeddings import SinusoidalPositionalEmbedding, TextEmbedding
from sopro.tokenizer import TextTokenizer

from .blocks import RMSNorm, SSMLiteBlock


class TextEncoder(nn.Module):
    def __init__(
        self, cfg: SoproTTSConfig, d_model: int, n_layers: int, tokenizer: TextTokenizer
    ):
        super().__init__()
        self.tok = tokenizer
        self.embed = TextEmbedding(self.tok.vocab_size, d_model)
        self.layers = nn.ModuleList(
            [SSMLiteBlock(d_model, cfg.dropout, causal=False) for _ in range(n_layers)]
        )
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=cfg.max_text_len + 8)
        self.norm = RMSNorm(d_model)

    def forward(
        self, text_ids: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(text_ids)
        L = x.size(1)
        pos = self.pos(torch.arange(L, device=x.device))
        x = x + pos.unsqueeze(0)

        x = x * mask.unsqueeze(-1).float()
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
        return x, pooled


class TextXAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0

        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = self.d_model // self.heads
        self.dropout = float(dropout)

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

        attn_mask = None
        if kpm is not None:
            kpm = kpm.to(torch.bool)

            keep = ~kpm

            bad = ~keep.any(dim=1)
            if bad.any():
                keep = keep.clone()
                keep[bad, 0] = True

            attn_mask = keep[:, None, None, :]

        with torch.autocast(device_type=x.device.type, enabled=False):
            a = F.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype)
        a = self.out_proj(self._from_heads(a))

        y = x + torch.tanh(self.gate) * a
        return (y, kv_cache) if use_cache else y
