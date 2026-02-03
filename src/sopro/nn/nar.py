from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from sopro.config import SoproTTSConfig

from .blocks import RMSNorm, SSMLiteBlock


class NARStageAdapter(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, stage_vec: torch.Tensor) -> torch.Tensor:
        if stage_vec.dim() == 1:
            stage_vec = stage_vec.unsqueeze(0).expand(x.size(0), -1)
        g, b = self.mlp(stage_vec).chunk(2, dim=-1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        x = self.norm(x)
        return x * (1 + torch.tanh(g)) + torch.tanh(b)


class NARSinglePass(nn.Module):
    def __init__(
        self, cfg: SoproTTSConfig, d_model: int, stage_specs: Dict[str, List[int]]
    ):
        super().__init__()
        self.cfg = cfg
        self.stage_names = [
            s for s in ["B", "C", "D", "E"] if len(stage_specs.get(s, [])) > 0
        ]
        self.stage_to_id = {s: i for i, s in enumerate(self.stage_names)}
        self.stage_specs = {s: stage_specs[s] for s in self.stage_names}

        ks = int(cfg.nar_kernel_size)
        cycle = tuple(int(x) for x in cfg.nar_dilation_cycle) or (1,)
        dils: List[int] = []
        while len(dils) < int(cfg.n_layers_nar):
            dils.extend(cycle)
        dils = dils[: int(cfg.n_layers_nar)]

        self.blocks = nn.ModuleList(
            [
                SSMLiteBlock(
                    d_model, cfg.dropout, causal=False, kernel_size=ks, dilation=int(d)
                )
                for d in dils
            ]
        )
        self.norm = RMSNorm(d_model)
        self.pre = nn.Linear(d_model, int(cfg.nar_head_dim))

        self.stage_emb = nn.Embedding(len(self.stage_names), d_model)
        self.adapter = NARStageAdapter(d_model, hidden=256)

        self.heads = nn.ModuleDict()
        self.head_id_emb = nn.ModuleDict()
        for s in self.stage_names:
            n_heads = len(self.stage_specs[s])
            self.heads[s] = nn.ModuleList(
                [
                    nn.Linear(int(cfg.nar_head_dim), int(cfg.codebook_size))
                    for _ in range(n_heads)
                ]
            )
            emb = nn.Embedding(n_heads, int(cfg.nar_head_dim))
            nn.init.zeros_(emb.weight)
            self.head_id_emb[s] = emb

        self.mix = nn.ParameterDict(
            {
                s: nn.Parameter(torch.zeros(2, dtype=torch.float32))
                for s in self.stage_names
            }
        )

    def forward_stage(
        self, stage: str, cond: torch.Tensor, prev_emb: torch.Tensor
    ) -> List[torch.Tensor]:
        if stage not in self.heads:
            return []

        w = torch.softmax(self.mix[stage], dim=0)
        x = w[0] * cond + w[1] * prev_emb

        sid = self.stage_to_id[stage]
        stage_vec = self.stage_emb.weight[sid]
        x = self.adapter(x, stage_vec)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = self.pre(x)
        outs: List[torch.Tensor] = []
        for i, head in enumerate(self.heads[stage]):
            hb = (
                self.head_id_emb[stage]
                .weight[i]
                .view(1, 1, -1)
                .to(dtype=z.dtype, device=z.device)
            )
            outs.append(head(z + hb))
        return outs
