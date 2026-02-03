from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthwiseConv1dState:
    buf: torch.Tensor


class GLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.pro = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.pro(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        y32 = x32 * torch.rsqrt(var + self.eps)
        y32 = y32 * self.weight.float()
        return y32.to(dtype=x.dtype)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self, d: int, kernel_size: int = 7, causal: bool = False, dilation: int = 1
    ):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.dw = nn.Conv1d(d, d, kernel_size, groups=d, padding=0, dilation=dilation)

    def _ctx_len(self) -> int:
        return (self.kernel_size - 1) * self.dilation + 1

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> DepthwiseConv1dState:
        if not self.causal:
            raise ValueError("init_state is only valid for causal convs")
        L = self._ctx_len()
        D = int(self.dw.in_channels)
        buf = torch.zeros((batch_size, L, D), device=device, dtype=dtype)
        return DepthwiseConv1dState(buf=buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        if self.causal:
            pad_left = (self.kernel_size - 1) * self.dilation
            xt = F.pad(xt, (pad_left, 0))
        else:
            total = (self.kernel_size - 1) * self.dilation
            left = total // 2
            right = total - left
            xt = F.pad(xt, (left, right))
        y = self.dw(xt)
        return y.transpose(1, 2)

    def forward_step(
        self, x_bt_d: torch.Tensor, state: Optional[DepthwiseConv1dState]
    ) -> Tuple[torch.Tensor, DepthwiseConv1dState]:
        if not self.causal:
            raise ValueError("forward_step is only valid for causal convs")

        if x_bt_d.dim() == 2:
            x_bt_d = x_bt_d.unsqueeze(1)

        B, T, D = x_bt_d.shape
        if T != 1:
            raise ValueError("forward_step expects a single timestep [B,1,D]")

        if state is None:
            state = self.init_state(B, x_bt_d.device, x_bt_d.dtype)

        buf = state.buf
        if buf.size(1) > 1:
            buf = torch.cat([buf[:, 1:, :], x_bt_d], dim=1)
        else:
            buf = x_bt_d

        k = int(self.kernel_size)
        d = int(self.dilation)
        idx = torch.arange(0, k * d, d, device=x_bt_d.device)
        x_bkd = buf.index_select(1, idx)  # [B,k,D]

        w_dk = self.dw.weight.squeeze(1).to(dtype=x_bt_d.dtype)
        y_bd = (x_bkd.transpose(1, 2) * w_dk.unsqueeze(0)).sum(dim=-1)
        if self.dw.bias is not None:
            y_bd = y_bd + self.dw.bias.to(dtype=y_bd.dtype).unsqueeze(0)

        y_bt_d = y_bd.unsqueeze(1)
        state.buf = buf
        return y_bt_d, state


class SSMLiteBlock(nn.Module):
    def __init__(
        self,
        d: int,
        dropout: float = 0.05,
        causal: bool = False,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm = RMSNorm(d)
        self.glu = GLU(d)
        self.dw = DepthwiseConv1d(
            d, kernel_size=kernel_size, causal=causal, dilation=dilation
        )
        self.ff = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> dict:
        if not self.dw.causal:
            raise ValueError("SSMLiteBlock.init_state only valid for causal blocks")
        return {"dw": self.dw.init_state(batch_size, device, dtype)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.glu(self.norm(x))
        h = self.dw(h)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(x))
        return x

    def forward_step(
        self, x_bt_d: torch.Tensor, state: dict
    ) -> Tuple[torch.Tensor, dict]:
        if not self.dw.causal:
            raise ValueError("forward_step only valid for causal blocks")

        h = self.glu(self.norm(x_bt_d))
        y, dw_state = self.dw.forward_step(h, state.get("dw", None))
        state["dw"] = dw_state

        x = x_bt_d + self.drop(y)
        x = x + self.drop(self.ff(x))
        return x, state


class AttentiveStatsPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, 1),
        )

    def forward(
        self, h: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = h.shape
        logits = self.attn(h).squeeze(-1)

        if lengths is not None:
            mask = torch.arange(T, device=h.device)[None, :] < lengths[:, None]
            logits = logits.masked_fill(~mask, -1e9)

        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        mu = (h * w).sum(dim=1)
        var = (w * (h - mu.unsqueeze(1)).pow(2)).sum(dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mu, std], dim=-1)
