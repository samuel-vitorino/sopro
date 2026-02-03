from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sopro.config import SoproTTSConfig
from sopro.nn.blocks import RMSNorm, SSMLiteBlock
from sopro.nn.text import TextXAttnBlock


class ARRVQ1Generator(nn.Module):
    def __init__(self, cfg: SoproTTSConfig, d_model: int, vocab: int):
        super().__init__()
        ks = int(cfg.ar_kernel)

        dils: List[int] = []
        while len(dils) < int(cfg.n_layers_ar):
            dils.extend(list(cfg.ar_dilation_cycle))
        dils = dils[: int(cfg.n_layers_ar)]
        self.dils = tuple(int(d) for d in dils)

        self.blocks = nn.ModuleList(
            [
                SSMLiteBlock(
                    d_model, cfg.dropout, causal=True, kernel_size=ks, dilation=d
                )
                for d in self.dils
            ]
        )

        self.attn_freq = int(cfg.ar_text_attn_freq)
        self.x_attns = nn.ModuleList()
        for i in range(len(self.blocks)):
            if (i + 1) % self.attn_freq == 0:
                self.x_attns.append(
                    TextXAttnBlock(d_model, heads=4, dropout=cfg.dropout)
                )
            else:
                self.x_attns.append(nn.Identity())

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    @torch.no_grad()
    def init_stream_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        layer_states = [
            blk.init_state(batch_size, device, dtype) for blk in self.blocks
        ]

        kv_caches: List[Optional[Dict[str, torch.Tensor]]] = []
        key_padding_mask = (~text_mask) if text_mask is not None else None
        for xa in self.x_attns:
            if isinstance(xa, nn.Identity) or (text_emb is None):
                kv_caches.append(None)
            else:
                kv_caches.append(
                    xa.build_kv_cache(text_emb, key_padding_mask=key_padding_mask)
                )

        return {"layer_states": layer_states, "kv_caches": kv_caches}

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = ~text_mask if text_mask is not None else None

        if key_padding_mask is not None:
            bad_rows = key_padding_mask.all(dim=1)
            if bad_rows.any():
                key_padding_mask = key_padding_mask.clone()
                idx = torch.nonzero(bad_rows, as_tuple=False).squeeze(1)
                key_padding_mask[idx, 0] = False
                if text_emb is not None:
                    text_emb = text_emb.clone()
                    text_emb[idx, 0, :] = 0

        h = x
        for i, lyr in enumerate(self.blocks):
            h = lyr(h)
            if not isinstance(self.x_attns[i], nn.Identity) and text_emb is not None:
                h = self.x_attns[i](h, text_emb, key_padding_mask=key_padding_mask)

        h = self.norm(h)

        return self.head(h)

    @torch.no_grad()
    def step(
        self,
        x_bt_d: torch.Tensor,
        state: Dict[str, object],
        *,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        h = x_bt_d
        key_padding_mask = (~text_mask) if text_mask is not None else None

        layer_states: List[dict] = state["layer_states"]
        kv_caches: List[Optional[Dict[str, torch.Tensor]]] = state["kv_caches"]

        for i, blk in enumerate(self.blocks):
            h, layer_states[i] = blk.forward_step(h, layer_states[i])

            xa = self.x_attns[i]
            if (not isinstance(xa, nn.Identity)) and (text_emb is not None):
                kv = kv_caches[i]
                if kv is None:
                    kv = xa.build_kv_cache(text_emb, key_padding_mask=key_padding_mask)
                h, kv = xa(h, kv_cache=kv, use_cache=True)
                kv_caches[i] = kv

        state["layer_states"] = layer_states
        state["kv_caches"] = kv_caches

        h = self.norm(h)
        logits = self.head(h)

        return logits, state
