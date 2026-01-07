from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe.index_select(0, positions.long())


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class CodebookEmbedding(nn.Module):
    def __init__(
        self, num_codebooks: int, vocab_size: int, d_model: int, use_bos: bool = True
    ):
        super().__init__()
        self.Q = int(num_codebooks)
        self.V = int(vocab_size)
        self.D = int(d_model)
        self.use_bos = bool(use_bos)

        table_size = self.Q * self.V + (1 if self.use_bos else 0)
        self.emb = nn.Embedding(table_size, d_model)
        self.bos_id = (self.Q * self.V) if self.use_bos else None

    def _indices_for(self, tokens: torch.Tensor, cb_index: int) -> torch.Tensor:
        return cb_index * self.V + tokens

    def embed_tokens(self, tokens: torch.Tensor, cb_index: int) -> torch.Tensor:
        return self.emb(self._indices_for(tokens, cb_index))

    def embed_shift_by_k(
        self, tokens: torch.Tensor, cb_index: int, k: int
    ) -> torch.Tensor:
        idx = self._indices_for(tokens, cb_index)
        B, T = idx.shape
        if (not self.use_bos) or (self.bos_id is None) or k <= 0:
            pad_tok = idx[:, :1]
        else:
            pad_tok = torch.full(
                (B, 1), self.bos_id, dtype=torch.long, device=idx.device
            )

        if k >= T:
            idx_shift = pad_tok.expand(-1, T)
        else:
            pad = pad_tok.expand(-1, k)
            idx_shift = torch.cat([pad, idx[:, :-k]], dim=1)

        return self.emb(idx_shift)

    def sum_embed_subset(
        self,
        tokens_subset: Optional[torch.Tensor],
        cb_indices: Optional[List[int]],
        keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens_subset is None or cb_indices is None or len(cb_indices) == 0:
            return 0.0

        B, T, K = tokens_subset.shape
        idx_list = []
        for k, cb in enumerate(cb_indices):
            idx_list.append(self._indices_for(tokens_subset[..., k], cb).unsqueeze(2))
        idx = torch.cat(idx_list, dim=2)
        emb = self.emb(idx)

        if keep_mask is not None:
            emb = emb * keep_mask.unsqueeze(-1).to(emb.dtype)

        return emb.sum(dim=2)
