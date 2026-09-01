from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sopro.config import ModelConfig
from sopro.nn.layers import apply_rotary, rotary_cos_sin, sinusoidal_time_embedding


def _expand_time(x: torch.Tensor, target_steps: int) -> torch.Tensor:
    src = int(x.shape[-1])
    idx = torch.div(torch.arange(int(target_steps), device=x.device) * src, int(target_steps), rounding_mode="floor").clamp_max(src - 1)
    return x.index_select(dim=2, index=idx)


def _row_has_signal(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(int(x.shape[0]), -1).abs().amax(dim=-1) > 0.0


def build_time_grid(steps: int, sway_coef: float, device: torch.device) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, steps=int(steps) + 1, device=device, dtype=torch.float32)
    if abs(float(sway_coef)) <= 1.0e-8:
        return times
    return times + (float(sway_coef) * (torch.cos(0.5 * math.pi * times) - 1.0 + times))


def build_chunk_mask(seq_len: int, chunk_size: int, num_left_chunks: int, device: torch.device) -> Optional[torch.Tensor]:
    if int(chunk_size) <= 0:
        return None
    positions = torch.arange(int(seq_len), device=device)
    chunk_index = torch.div(positions, int(chunk_size), rounding_mode="floor")
    chunk_end = ((chunk_index + 1) * int(chunk_size)).clamp_max(int(seq_len))
    if int(num_left_chunks) < 0:
        chunk_start = torch.zeros_like(chunk_end)
    else:
        chunk_start = ((chunk_index - int(num_left_chunks)).clamp_min(0) * int(chunk_size)).clamp_max(int(seq_len))
    key_pos = positions.unsqueeze(0)
    return ((key_pos >= chunk_start.unsqueeze(1)) & (key_pos < chunk_end.unsqueeze(1))).unsqueeze(0)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.left_context = int(kernel_size) - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_context, 0)))


class LearnedCausalUpsampler(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        hidden = max(8, int(channels))
        self.in_proj = nn.Conv1d(int(channels), hidden, kernel_size=1)
        self.mix = CausalConv1d(hidden, hidden, kernel_size=int(kernel_size))
        self.out_proj = nn.Conv1d(hidden, int(channels), kernel_size=1)

    def forward(self, x: torch.Tensor, target_steps: int) -> torch.Tensor:
        x_rep = _expand_time(x, target_steps)
        h = F.silu(self.in_proj(x_rep))
        h = F.silu(self.mix(h))
        return x_rep + self.out_proj(h)


class PreLookahead(nn.Module):
    def __init__(self, channels: int, lookahead: int) -> None:
        super().__init__()
        self.lookahead = int(lookahead)
        self.conv1 = nn.Conv1d(int(channels), int(channels), kernel_size=self.lookahead + 1)
        self.conv2 = nn.Conv1d(int(channels), int(channels), kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (0, self.lookahead))
        y = F.leaky_relu(self.conv1(y), negative_slope=0.1)
        y = F.pad(y, (2, 0))
        return x + self.conv2(y)


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int, groups: int = 16) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv1 = nn.Conv1d(int(dim), int(dim), self.kernel_size, groups=int(groups))
        self.conv2 = nn.Conv1d(int(dim), int(dim), self.kernel_size, groups=int(groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2).contiguous()
        y = F.mish(self.conv1(F.pad(y, (self.kernel_size - 1, 0))))
        y = F.mish(self.conv2(F.pad(y, (self.kernel_size - 1, 0))))
        return y.transpose(1, 2).contiguous()


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), int(dim) * 6))

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(emb).chunk(6, dim=-1)
        return self.norm(x) * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :], gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), int(dim) * 2))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiTSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner = self.heads * self.dim_head
        self.to_q = nn.Linear(int(dim), inner)
        self.to_k = nn.Linear(int(dim), inner)
        self.to_v = nn.Linear(int(dim), inner)
        self.to_out = nn.Sequential(nn.Linear(inner, int(dim)), nn.Identity())

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.to_q(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        k = self.to_k(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        v = self.to_v(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if attn_mask is None else attn_mask[:, None, :, :])
        return self.to_out(y.transpose(1, 2).contiguous().view(bsz, seq_len, self.heads * self.dim_head))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNormZero(int(dim))
        self.attn = DiTSelfAttention(int(dim), int(heads), int(dim_head))
        self.ff_norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-6)
        hidden = max(1, int(round(float(ff_mult) * int(dim))))
        self.ff = nn.Sequential(nn.Linear(int(dim), hidden), nn.GELU(approximate="tanh"), nn.Identity(), nn.Linear(hidden, int(dim)))

    def forward(self, x: torch.Tensor, emb: torch.Tensor, attn_mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb)
        x = x + gate_msa[:, None, :] * self.attn(h, attn_mask, cos, sin)
        h_ff = self.ff_norm(x) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return x + gate_mlp[:, None, :] * self.ff(h_ff)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, mu_dim: int, spk_dim: int, out_dim: int, pos_kernel_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear((int(mel_dim) * 2) + int(mu_dim) + int(spk_dim), int(out_dim))
        self.pos = CausalConvPositionEmbedding(int(out_dim), kernel_size=int(pos_kernel_size))
        self.cond_mask_proj = nn.Linear(1, int(out_dim), bias=False)

    def forward(self, x_t: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, mu: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x_btc = x_t.transpose(1, 2).contiguous()
        cond_btc = cond_mel.transpose(1, 2).contiguous()
        cond_mask_btc = cond_mask.transpose(1, 2).contiguous().to(dtype=x_btc.dtype)
        mu_btc = mu.transpose(1, 2).contiguous()
        spk_btc = spk[:, None, :].expand(-1, x_btc.shape[1], -1)
        h = self.proj(torch.cat([x_btc, cond_btc, mu_btc, spk_btc], dim=-1))
        h = h + self.cond_mask_proj(cond_mask_btc)
        return h + self.pos(h)


class AcousticHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.semantic_dim = int(cfg.latent_dim)
        self.target_dim = int(cfg.acoustic_mel_n_mels)
        self.model_dim = int(cfg.acoustic_dit_dim)
        self.time_embed_dim = int(cfg.acoustic_time_embed_dim)
        self.mu_dim = int(cfg.acoustic_mu_dim)
        self.spk_dim = int(cfg.acoustic_spk_dim)
        self.dim_head = int(cfg.acoustic_dit_dim_head)
        self.semantic_token_emb = nn.Embedding(int(cfg.semantic_vocab_size), self.semantic_dim)
        self.semantic_prelook = PreLookahead(self.semantic_dim, int(cfg.acoustic_pre_lookahead_frames))
        self.semantic_upsampler = LearnedCausalUpsampler(self.semantic_dim, int(cfg.acoustic_upsampler_kernel_size))
        self.mu_proj = nn.Conv1d(self.semantic_dim, self.mu_dim, kernel_size=1)
        self.time_mlp = nn.Sequential(nn.Linear(self.time_embed_dim, self.model_dim), nn.SiLU(), nn.Linear(self.model_dim, self.model_dim))
        self.spk_proj = nn.Linear(int(cfg.cond_hidden_dim), self.spk_dim)
        self.input_embed = InputEmbedding(self.target_dim, self.mu_dim, self.spk_dim, self.model_dim, int(cfg.acoustic_pos_kernel_size))
        self.blocks = nn.ModuleList(
            [DiTBlock(self.model_dim, int(cfg.acoustic_dit_heads), self.dim_head, float(cfg.acoustic_dit_ff_mult)) for _ in range(int(cfg.acoustic_dit_depth))]
        )
        self.out_norm = AdaLayerNormFinal(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.target_dim)

    def semantic_latents(self, semantic_tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.semantic_token_emb(semantic_tokens.to(dtype=torch.long)).transpose(1, 2).contiguous().to(dtype=dtype)

    def _time_embedding(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t.to(dtype=torch.float32), self.time_embed_dim)
        return self.time_mlp(emb.to(dtype=self.time_mlp[0].weight.dtype)).to(dtype=dtype)

    def _speaker_embedding(self, cond_vec: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        cond = F.normalize(cond_vec.to(dtype=self.spk_proj.weight.dtype), dim=-1)
        spk = self.spk_proj(cond)
        return (spk * _row_has_signal(cond_vec).to(dtype=spk.dtype)[:, None]).to(dtype=dtype)

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        semantic_latents: torch.Tensor,
        cond_vec: torch.Tensor,
        cond_mel: torch.Tensor,
        cond_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        mu = self.mu_proj(self.semantic_upsampler(self.semantic_prelook(semantic_latents), int(x_t.shape[-1])))
        mu = mu * _row_has_signal(semantic_latents)[:, None, None].to(dtype=mu.dtype)
        spk = self._speaker_embedding(cond_vec, x_t.dtype)
        emb = self._time_embedding(t, x_t.dtype)
        h = self.input_embed(x_t, cond_mel, cond_mask, mu, spk)
        for block in self.blocks:
            h = block(h, emb, attn_mask, cos, sin)
        h = self.out_norm(h, emb)
        return self.out_proj(h).transpose(1, 2).contiguous()

    @torch.no_grad()
    def solve(
        self,
        x_init: torch.Tensor,
        semantic_tokens: torch.Tensor,
        cond_vec: torch.Tensor,
        cond_mel: torch.Tensor,
        cond_mask: torch.Tensor,
        steps: int,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        x = x_init
        seq_len = int(x.shape[-1])
        cond_mask = cond_mask.to(dtype=x.dtype).clamp(0.0, 1.0)
        cond_mel = cond_mel.to(dtype=x.dtype) * cond_mask
        sigma_min = float(self.cfg.acoustic_sigma_min)
        grid = build_time_grid(int(steps), float(self.cfg.acoustic_sway_sampling_coef), x.device)
        latents = self.semantic_latents(semantic_tokens, x.dtype)
        attn_mask = build_chunk_mask(seq_len, int(chunk_size), int(self.cfg.acoustic_num_left_chunks), x.device)
        cos, sin = rotary_cos_sin(torch.arange(seq_len, device=x.device), self.dim_head, x.dtype)
        batch = int(x.shape[0])
        for i in range(int(steps)):
            t0, t1 = grid[i], grid[i + 1]
            v = self.velocity(x, t0.expand(batch), latents, cond_vec, cond_mel, cond_mask, attn_mask, cos, sin)
            x = x + ((t1 - t0) * v)
            x_prompt = (1.0 - (1.0 - sigma_min) * t1) * x_init + t1 * cond_mel
            x = (cond_mask * x_prompt) + ((1.0 - cond_mask) * x)
        return (cond_mask * cond_mel) + ((1.0 - cond_mask) * x)

    def new_chunked_state(self, steps: int, max_frames: int) -> "ChunkedSolveState":
        return ChunkedSolveState(int(steps), len(self.blocks), int(max_frames))

    def _embed_window(self, x_win: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, mu: torch.Tensor, spk: torch.Tensor, s0: int, c: int) -> torch.Tensor:
        ie = self.input_embed
        n = int(x_win.shape[-1])
        h = ie.proj(torch.cat([x_win.transpose(1, 2), cond_mel.transpose(1, 2), mu.transpose(1, 2), spk[:, None, :].expand(-1, n, -1)], dim=-1))
        h = h + ie.cond_mask_proj(cond_mask.transpose(1, 2).to(dtype=h.dtype))
        y = h.transpose(1, 2).contiguous()
        k = ie.pos.kernel_size
        if s0 == 0:
            y = F.mish(ie.pos.conv1(F.pad(y, (k - 1, 0))))
            y = F.mish(ie.pos.conv2(F.pad(y, (k - 1, 0))))[:, :, c:]
        else:
            y = F.mish(ie.pos.conv2(F.mish(ie.pos.conv1(y))))
        return h[:, c - s0 :] + y.transpose(1, 2).contiguous()

    def _prelook_window(self, latents: torch.Tensor, a: int, b: int) -> torch.Tensor:
        pl = self.semantic_prelook
        n_tok = int(latents.shape[-1])
        lo, hi = max(0, a - 2), min(n_tok, b + pl.lookahead)
        y = latents[:, :, lo:hi]
        if b + pl.lookahead > n_tok:
            y = F.pad(y, (0, b + pl.lookahead - n_tok))
        y = F.leaky_relu(pl.conv1(y), negative_slope=0.1)
        if a - lo < 2:
            y = F.pad(y, (2 - (a - lo), 0))
        return latents[:, :, a:b] + pl.conv2(y)

    def _extend_mu(self, state: "ChunkedSolveState", latents: torch.Tensor, T: int, t_full: int) -> None:
        if T <= state.n_mu:
            return
        n_tok = int(latents.shape[-1])
        need_tok = min(n_tok, ((T - 1) * n_tok) // t_full + 1)
        n_final = max(0, n_tok - self.semantic_prelook.lookahead)
        if n_final > state.n_prelook:
            block = self._prelook_window(latents, state.n_prelook, n_final)
            state.prelook = block if state.prelook is None else torch.cat([state.prelook, block], dim=-1)
            state.n_prelook = n_final
        prelook = state.prelook
        if need_tok > state.n_prelook:
            tail = self._prelook_window(latents, state.n_prelook, need_tok)
            prelook = tail if prelook is None else torch.cat([prelook, tail], dim=-1)
        up = self.semantic_upsampler
        k = up.mix.left_context
        f0, f1 = state.n_mu, T
        lo = max(0, f0 - k)
        idx = torch.div(torch.arange(lo, f1, device=latents.device) * n_tok, t_full, rounding_mode="floor").clamp_max(n_tok - 1)
        x_rep = prelook.index_select(2, idx)
        h = F.silu(up.in_proj(x_rep))
        if f0 - lo < k:
            h = F.pad(h, (k - (f0 - lo), 0))
        h = F.silu(up.mix.conv(h))
        mu = self.mu_proj(x_rep[:, :, f0 - lo :] + up.out_proj(h))
        state.mu[:, :, f0:f1] = mu * _row_has_signal(latents)[:, None, None].to(dtype=mu.dtype)
        state.n_mu = T

    @torch.no_grad()
    def solve_chunked(
        self,
        state: "ChunkedSolveState",
        x0: torch.Tensor,
        semantic_tokens: torch.Tensor,
        cond_vec: torch.Tensor,
        cond_mel: torch.Tensor,
        cond_mask: torch.Tensor,
        chunk_size: int,
        keep_end: int,
    ) -> torch.Tensor:
        steps = state.steps
        t_full = int(x0.shape[-1])
        c = state.cached
        T = min(int(keep_end), t_full)
        n_keep = T - c
        if n_keep <= 0:
            return x0.new_zeros((1, int(x0.shape[1]), 0))
        dtype = x0.dtype
        cond_mask = cond_mask.to(dtype=dtype).clamp(0.0, 1.0)
        cond_mel = cond_mel.to(dtype=dtype) * cond_mask
        sigma_min = float(self.cfg.acoustic_sigma_min)
        grid = build_time_grid(steps, float(self.cfg.acoustic_sway_sampling_coef), x0.device)
        latents = self.semantic_latents(semantic_tokens, dtype)
        spk = self._speaker_embedding(cond_vec, dtype)
        s0 = max(0, c - 2 * (self.input_embed.pos.kernel_size - 1))
        heads, dim_head = self.blocks[0].attn.heads, self.blocks[0].attn.dim_head
        state.ensure(T, int(x0.shape[1]), self.mu_dim, heads, dim_head, x0.device, dtype)
        self._extend_mu(state, latents, T, t_full)
        mu = state.mu
        positions = torch.arange(T, device=x0.device)
        cos, sin = rotary_cos_sin(positions, self.dim_head, dtype)
        cos_q, sin_q = cos[c:T], sin[c:T]
        chunk_index = torch.div(positions[c:T], int(chunk_size), rounding_mode="floor")
        chunk_end = ((chunk_index + 1) * int(chunk_size)).clamp_max(T)
        left = int(self.cfg.acoustic_num_left_chunks)
        chunk_start = torch.zeros_like(chunk_end) if left < 0 else ((chunk_index - left).clamp_min(0) * int(chunk_size)).clamp_max(T)
        attn_mask = ((positions.unsqueeze(0) >= chunk_start.unsqueeze(1)) & (positions.unsqueeze(0) < chunk_end.unsqueeze(1)))[None, None, :, :]
        cond_mel_new, cond_mask_new = cond_mel[:, :, c:T], cond_mask[:, :, c:T]
        x0_new = x0[:, :, c:T]
        x_new = x0_new
        for i in range(steps):
            t0, t1 = grid[i], grid[i + 1]
            if state.emb[i] is None:
                state.emb[i] = self._time_embedding(t0.expand(1), dtype)
            emb = state.emb[i]
            x_win = torch.cat([state.x[i][:, :, s0:c], x_new], dim=-1) if c > s0 else x_new
            h = self._embed_window(x_win, cond_mel[:, :, s0:T], cond_mask[:, :, s0:T], mu[:, :, s0:T], spk, s0, c)
            n = int(h.shape[1])
            for l, block in enumerate(self.blocks):
                hn, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb)
                attn = block.attn
                q = apply_rotary(attn.to_q(hn).view(1, n, attn.heads, attn.dim_head).transpose(1, 2).contiguous(), cos_q, sin_q)
                k = apply_rotary(attn.to_k(hn).view(1, n, attn.heads, attn.dim_head).transpose(1, 2).contiguous(), cos_q, sin_q)
                v = attn.to_v(hn).view(1, n, attn.heads, attn.dim_head).transpose(1, 2).contiguous()
                k_buf, v_buf = state.kv[i][l]
                k_buf[:, :, c:T] = k
                v_buf[:, :, c:T] = v
                y = F.scaled_dot_product_attention(q, k_buf[:, :, :T], v_buf[:, :, :T], attn_mask=attn_mask)
                h = h + gate_msa[:, None, :] * attn.to_out(y.transpose(1, 2).contiguous().view(1, n, attn.heads * attn.dim_head))
                h = h + gate_mlp[:, None, :] * block.ff(block.ff_norm(h) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :])
            vel = self.out_proj(self.out_norm(h, emb)).transpose(1, 2).contiguous()
            state.x[i][:, :, c:T] = x_new
            x_new = x_new + ((t1 - t0) * vel)
            x_prompt = (1.0 - (1.0 - sigma_min) * t1) * x0_new + t1 * cond_mel_new
            x_new = (cond_mask_new * x_prompt) + ((1.0 - cond_mask_new) * x_new)
        out = (cond_mask_new * cond_mel_new) + ((1.0 - cond_mask_new) * x_new)
        state.cached = T
        return out


class ChunkedSolveState:
    def __init__(self, steps: int, num_layers: int, max_frames: int) -> None:
        self.steps = int(steps)
        self.num_layers = int(num_layers)
        self.capacity = max(1, int(max_frames))
        self.x: list = [None] * self.steps
        self.kv: list = [[None] * self.num_layers for _ in range(self.steps)]
        self.emb: list = [None] * self.steps
        self.cached = 0
        self.mu: Optional[torch.Tensor] = None
        self.n_mu = 0
        self.prelook: Optional[torch.Tensor] = None
        self.n_prelook = 0

    def ensure(self, frames: int, n_mels: int, mu_dim: int, heads: int, dim_head: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.x[0] is not None and int(frames) <= self.capacity:
            return
        self._realloc(max(self.capacity, int(frames)), int(n_mels), int(mu_dim), int(heads), int(dim_head), device, dtype, self)

    def _realloc(self, capacity: int, n_mels: int, mu_dim: int, heads: int, dim_head: int, device: torch.device, dtype: torch.dtype, src: "ChunkedSolveState") -> None:
        n = int(src.cached)
        mu = torch.zeros((1, mu_dim, capacity), device=device, dtype=dtype)
        if src.mu is not None:
            mu[:, :, : src.n_mu] = src.mu[:, :, : src.n_mu]
        self.mu = mu
        for i in range(self.steps):
            x = torch.zeros((1, n_mels, capacity), device=device, dtype=dtype)
            if src.x[i] is not None:
                x[:, :, :n] = src.x[i][:, :, :n]
            self.x[i] = x
            for l in range(self.num_layers):
                k = torch.zeros((1, heads, capacity, dim_head), device=device, dtype=dtype)
                v = torch.zeros((1, heads, capacity, dim_head), device=device, dtype=dtype)
                if src.kv[i][l] is not None:
                    k[:, :, :n] = src.kv[i][l][0][:, :, :n]
                    v[:, :, :n] = src.kv[i][l][1][:, :, :n]
                self.kv[i][l] = (k, v)
        self.capacity = capacity

    def expanded(self, capacity: int) -> "ChunkedSolveState":
        out = ChunkedSolveState(self.steps, self.num_layers, capacity)
        out.cached, out.n_mu, out.n_prelook = self.cached, self.n_mu, self.n_prelook
        out.prelook, out.emb = self.prelook, list(self.emb)
        k0 = self.kv[0][0][0]
        out._realloc(max(int(capacity), self.capacity), int(self.x[0].shape[1]), int(self.mu.shape[1]), int(k0.shape[1]), int(k0.shape[-1]), k0.device, k0.dtype, self)
        return out

    def to(self, device: torch.device) -> "ChunkedSolveState":
        out = ChunkedSolveState(self.steps, self.num_layers, self.capacity)
        out.cached, out.n_mu, out.n_prelook = self.cached, self.n_mu, self.n_prelook
        out.prelook = None if self.prelook is None else self.prelook.to(device)
        out.emb = [None if e is None else e.to(device) for e in self.emb]
        out.mu = None if self.mu is None else self.mu.to(device)
        out.x = [None if x is None else x.to(device) for x in self.x]
        out.kv = [[None if kv is None else (kv[0].to(device), kv[1].to(device)) for kv in row] for row in self.kv]
        return out
