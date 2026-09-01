from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sopro.config import VocoderConfig


class MelFeatures(nn.Module):
    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(cfg.sample_rate), n_fft=int(cfg.n_fft), hop_length=int(cfg.hop_length), n_mels=int(cfg.n_mels), center=True, power=1
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clip(self.mel_spec(audio), min=1e-7))


def _conv_causal(x: torch.Tensor, conv: nn.Conv1d, lookahead: int) -> torch.Tensor:
    context = int(conv.kernel_size[0]) - 1
    return F.conv1d(F.pad(x, (context - int(lookahead), int(lookahead))), conv.weight, conv.bias, groups=conv.groups)


def _depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    k = int(weight.shape[-1])
    n = int(x.shape[-1]) - k + 1
    out = x[:, :, :n] * weight[None, :, 0, 0:1]
    for i in range(1, k):
        out = out + x[:, :, i : i + n] * weight[None, :, 0, i : i + 1]
    return out + bias[None, :, None]


def _conv_causal_stream(x: torch.Tensor, conv: nn.Conv1d, state: Optional[torch.Tensor], lookahead: int, flush: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    context = int(conv.kernel_size[0]) - 1
    right = int(lookahead)
    if flush and right > 0:
        x = F.pad(x, (0, right))
    if state is None:
        state = x.new_zeros(int(x.shape[0]), int(x.shape[1]), context - right)
    available = torch.cat([state, x], dim=-1)
    n_out = int(available.shape[-1]) - context
    if n_out <= 0:
        return x.new_zeros(int(x.shape[0]), int(conv.weight.shape[0]), 0), available.contiguous()
    if int(conv.groups) == int(conv.weight.shape[0]) and int(conv.weight.shape[1]) == 1:
        y = _depthwise_conv1d(available, conv.weight, conv.bias)
    else:
        y = F.conv1d(available, conv.weight, conv.bias, groups=conv.groups)
    return y, available[:, :, -context:].contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float, causal: bool, lookahead: int = 0) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.lookahead = int(lookahead)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=0 if causal else 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def _pointwise(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x.transpose(1, 2)))))
        return (self.gamma * x).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _conv_causal(x, self.dwconv, self.lookahead) if self.causal else self.dwconv(x)
        return x + self._pointwise(h)

    def forward_stream(self, x: torch.Tensor, state: Optional[Dict[str, Any]], flush: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = state or {"conv": None, "pending": None}
        pending = state["pending"]
        if pending is None:
            pending = x.new_zeros(int(x.shape[0]), int(x.shape[1]), 0)
        residual_stream = torch.cat([pending, x], dim=-1)
        y, conv_state = _conv_causal_stream(x, self.dwconv, state["conv"], self.lookahead, flush)
        n_out = int(y.shape[-1])
        residual = residual_stream[:, :, :n_out]
        return residual + self._pointwise(y), {"conv": conv_state, "pending": residual_stream[:, :, n_out:].contiguous()}


class Backbone(nn.Module):
    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.causal = bool(cfg.causal)
        self.lookahead_frames = int(cfg.lookahead_frames) if self.causal else 0
        block_lookaheads = list(cfg.block_lookaheads) if (self.causal and cfg.block_lookaheads) else [0] * int(cfg.num_layers)
        self.block_lookaheads = [int(v) for v in block_lookaheads]
        self.embed = nn.Conv1d(int(cfg.n_mels), int(cfg.dim), kernel_size=7, padding=0 if self.causal else 3)
        self.norm = nn.LayerNorm(int(cfg.dim), eps=1e-6)
        self.convnext = nn.ModuleList(
            [ConvNeXtBlock(int(cfg.dim), int(cfg.intermediate_dim), 1.0 / int(cfg.num_layers), self.causal, self.block_lookaheads[i]) for i in range(int(cfg.num_layers))]
        )
        self.final_layer_norm = nn.LayerNorm(int(cfg.dim), eps=1e-6)

    @property
    def total_lookahead(self) -> int:
        return self.lookahead_frames + sum(self.block_lookaheads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _conv_causal(x, self.embed, self.lookahead_frames) if self.causal else self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))

    def forward_stream(self, x: torch.Tensor, state: Optional[Dict[str, Any]], flush: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = state or {"embed": None, "blocks": [None] * len(self.convnext)}
        x, embed_state = _conv_causal_stream(x, self.embed, state["embed"], self.lookahead_frames, flush)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        block_states = []
        for i, block in enumerate(self.convnext):
            x, s = block.forward_stream(x, state["blocks"][i], flush)
            block_states.append(s)
        return self.final_layer_norm(x.transpose(1, 2)), {"embed": embed_state, "blocks": block_states}


@dataclass
class ISTFTState:
    processed_frames: int = 0
    emitted_samples: int = 0
    tail_start: int = 0
    ola: Optional[torch.Tensor] = None
    env: Optional[torch.Tensor] = None


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft))

    @property
    def pad(self) -> int:
        return self.n_fft // 2

    def _overlap_add(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, t = spec.shape
        if t <= 0:
            return spec.real.new_zeros(b, 0), self.window.new_zeros(0)
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        output_size = (t - 1) * self.hop_length + self.n_fft
        y = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))[:, 0, 0, :]
        window_sq = self.window.square().view(1, self.n_fft, 1).expand(1, self.n_fft, t)
        env = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))[0, 0, 0, :]
        return y, env

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        y, env = self._overlap_add(spec)
        y = y[:, self.pad : -self.pad]
        env = env[self.pad : -self.pad]
        return y / env.unsqueeze(0)

    def forward_stream(self, spec: torch.Tensor, state: Optional[ISTFTState], flush: bool) -> Tuple[torch.Tensor, ISTFTState]:
        st = state or ISTFTState()
        b = int(spec.shape[0])
        dev, dt = spec.real.device, spec.real.dtype
        y_chunk, env_chunk = self._overlap_add(spec)
        offset = st.processed_frames * self.hop_length - st.tail_start
        required = offset + int(y_chunk.shape[-1])
        cur = 0 if st.ola is None else int(st.ola.shape[-1])
        length = max(cur, required)
        ola = torch.zeros(b, length, device=dev, dtype=dt)
        env = torch.zeros(length, device=dev, dtype=dt)
        if cur > 0:
            ola[:, :cur] = st.ola
            env[:cur] = st.env
        if int(y_chunk.shape[-1]) > 0:
            ola[:, offset:required] += y_chunk
            env[offset:required] += env_chunk
        st.processed_frames += int(spec.shape[-1])
        target = max(0, st.processed_frames * self.hop_length - self.pad)
        if flush and st.processed_frames > 0:
            target = max(target, (st.processed_frames - 1) * self.hop_length + self.n_fft - 2 * self.pad)
        emit_count = max(0, target - st.emitted_samples)
        rel_start = st.emitted_samples + self.pad - st.tail_start
        rel_end = rel_start + emit_count
        out = ola[:, rel_start:rel_end] / env[rel_start:rel_end].clamp_min(1.0e-8).unsqueeze(0)
        st.emitted_samples = target
        trim = st.emitted_samples + self.pad - st.tail_start
        st.ola = ola[:, trim:].contiguous()
        st.env = env[trim:].contiguous()
        st.tail_start += trim
        return out, (ISTFTState() if flush else st)


class ISTFTHead(nn.Module):
    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.out = nn.Linear(int(cfg.dim), int(cfg.n_fft) + 2)
        self.istft = ISTFT(int(cfg.n_fft), int(cfg.hop_length))
        self.log_max_magnitude = math.log(float(cfg.max_magnitude))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        mag, p = self.out(x).transpose(1, 2).chunk(2, dim=1)
        mag = torch.exp(torch.clamp(mag, max=self.log_max_magnitude))
        return mag * (torch.cos(p) + 1j * torch.sin(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.istft(self.spectrogram(x))

    def forward_stream(self, x: torch.Tensor, state: Optional[ISTFTState], flush: bool) -> Tuple[torch.Tensor, ISTFTState]:
        return self.istft.forward_stream(self.spectrogram(x), state, flush)


class Vocoder(nn.Module):
    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = MelFeatures(cfg)
        self.backbone = Backbone(cfg)
        self.head = ISTFTHead(cfg)

    @property
    def hop_length(self) -> int:
        return int(self.cfg.hop_length)

    @torch.no_grad()
    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(audio.float())

    @torch.no_grad()
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(mel.float()))

    @torch.no_grad()
    def decode_stream(self, mel: torch.Tensor, state: Optional[Dict[str, Any]], flush: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = state or {"backbone": None, "istft": None}
        h, backbone_state = self.backbone.forward_stream(mel.float(), state["backbone"], flush)
        audio, istft_state = self.head.forward_stream(h, state["istft"], flush)
        return audio, {"backbone": backbone_state, "istft": istft_state}
