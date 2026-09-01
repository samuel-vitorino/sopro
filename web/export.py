#!/usr/bin/env python3
"""Export the shipping Sopro package to browser-oriented ONNX graphs.

This script deliberately lives outside ``src/sopro``.  Its dependencies are
export-time tools only and never become dependencies of the Python wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from sopro.config import SoproConfig  # noqa: E402
from sopro.encoders.semantic import SemanticEncoder  # noqa: E402
from sopro.encoders.speaker import SpeakerEncoder  # noqa: E402
from sopro.hub import (  # noqa: E402
    MODEL_FILE,
    SEMANTIC_ENCODER_FILE,
    SPEAKER_ENCODER_FILE,
    TOKENIZER_FILE,
    VOCODER_FILE,
    load_config,
    load_weights,
)
from sopro.nn.layers import apply_rotary  # noqa: E402
from sopro.nn.model import SoproModel  # noqa: E402
from sopro.vocoder import Vocoder  # noqa: E402

OPSET = 20
POS_CONTEXT = 60
ACOUSTIC_STEPS = 2


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    if bias is not None:
        scores = scores + bias.float()
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(dtype=q.dtype)


def _ar_layer(layer: nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_k: torch.Tensor | None = None, past_v: torch.Tensor | None = None, bias: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = layer.attn
    h = layer.attn_norm(x)
    b, t, _ = h.shape
    q = attn.q_proj(h).view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h).view(b, t, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h).view(b, t, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    if attn.qk_rms_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    present_k = k if past_k is None else torch.cat([past_k, k], dim=2)
    present_v = v if past_v is None else torch.cat([past_v, v], dim=2)
    keys, values = present_k, present_v
    if attn.num_kv_groups > 1:
        keys = keys.repeat_interleave(attn.num_kv_groups, dim=1)
        values = values.repeat_interleave(attn.num_kv_groups, dim=1)
    y = _attention(q, keys, values, bias)
    y = attn.out_proj(y.transpose(1, 2).contiguous().view(b, t, attn.model_dim))
    x = x + layer.attn_scale(y)
    x = x + layer.ffn_scale(layer.ffn(layer.ffn_norm(x)))
    return x, present_k, present_v


class ReferenceGraph(nn.Module):
    def __init__(self, model: SoproModel, semantic: SemanticEncoder, speaker: SpeakerEncoder) -> None:
        super().__init__()
        self.cond_proj = model.cond_proj
        self.semantic = semantic
        self.speaker = speaker

    def forward(self, semantic_mel: torch.Tensor, interp_left: torch.Tensor, interp_right: torch.Tensor, interp_weight: torch.Tensor, speaker_mel: torch.Tensor):
        sem = self.semantic
        x = F.gelu(sem.conv1(semantic_mel))
        x = F.gelu(sem.conv2(x)).permute(0, 2, 1)
        x = x + sem.pos_emb[: x.shape[1]].to(x.dtype).unsqueeze(0)
        for layer in sem.layers:
            x = layer(x)
        x = sem.final_norm(x)
        weight = interp_weight.view(1, -1, 1).to(dtype=x.dtype)
        x = x.index_select(1, interp_left) * (1.0 - weight) + x.index_select(1, interp_right) * weight
        logits = sem.digit_head(sem.pre_head_norm(x))
        digits = torch.stack([piece.argmax(dim=-1) for piece in torch.split(logits, sem.levels, dim=-1)], dim=-1)
        semantic_tokens = (digits.to(torch.long) * sem._bases.view(1, 1, -1)).sum(dim=-1)

        spk = self.speaker
        y = spk.stem(speaker_mel)
        feats = []
        for transition, stage in zip(spk.transitions, spk.stages):
            y = stage(transition(y))
            feats.append(y)
        y = spk.fuse(torch.cat(feats, dim=1))
        id_emb = F.normalize(spk.id_head(spk.id_pool(y)), p=2, dim=-1)
        local = spk.style_pool.local_pool(y)
        pooled_parts = []
        for value in (y, local):
            value_mean = value.mean(dim=-1)
            value_std = torch.sqrt(((value - value_mean.unsqueeze(-1)) ** 2).mean(dim=-1).clamp_min(1.0e-6))
            pooled_parts.extend([value_mean, value_std])
        pooled = torch.cat(pooled_parts, dim=1)
        cond = self.cond_proj(torch.cat([id_emb, spk.style_head(pooled), spk.style_ctrl_head(pooled)], dim=-1))
        return cond, semantic_tokens


class SemanticPrefillGraph(nn.Module):
    def __init__(self, model: SoproModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, text_ids: torch.Tensor, style_tokens: torch.Tensor, prompt_tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal_bias: torch.Tensor):
        prefix = self.model.build_prefix(text_ids, style_tokens, prompt_tokens)
        h, keys, values = prefix, [], []
        for layer in self.model.ar_prior.temporal.layers:
            h, k, v = _ar_layer(layer, h, cos, sin, bias=causal_bias)
            keys.append(k)
            values.append(v)
        logits = self.model.ar_prior.token_head(self.model.ar_prior.out_norm(h[:, -1:, :]))[:, 0]
        return logits, torch.stack(keys), torch.stack(values)


class SemanticStepGraph(nn.Module):
    def __init__(self, model: SoproModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, token: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        h = self.model.embed_semantic(token)
        keys, values = [], []
        for index, layer in enumerate(self.model.ar_prior.temporal.layers):
            h, k, v = _ar_layer(layer, h, cos, sin, past_k[index], past_v[index])
            keys.append(k)
            values.append(v)
        logits = self.model.ar_prior.token_head(self.model.ar_prior.out_norm(h[:, -1:, :]))[:, 0]
        return logits, torch.stack(keys), torch.stack(values)


class SemanticPrefixEmbeddingGraph(nn.Module):
    def __init__(self, model: SoproModel) -> None:
        super().__init__()
        self.model = model
        self.bos_id = int(model.cfg.semantic_bos_id)

    def forward(self, text_ids: torch.Tensor, style_tokens: torch.Tensor, prompt_tokens: torch.Tensor):
        model = self.model
        bos = torch.full((1, 1), self.bos_id, device=text_ids.device, dtype=torch.long)
        bos_hidden = model.embed_semantic(bos)
        style = model.style_prefix(model.embed_semantic(style_tokens)).to(dtype=bos_hidden.dtype)
        text = model.text_tok_emb(text_ids).to(dtype=bos_hidden.dtype)
        prompt = model.embed_semantic(prompt_tokens).to(dtype=bos_hidden.dtype)
        return torch.cat([style, text, prompt], dim=1), bos_hidden


class SemanticCoreGraph(nn.Module):
    """One transformer used for both full-prefix prefill and cached steps."""

    def __init__(self, model: SoproModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, hidden: torch.Tensor, token: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_bias: torch.Tensor):
        h = torch.cat([hidden, self.model.embed_semantic(token)], dim=1)
        keys, values = [], []
        for index, layer in enumerate(self.model.ar_prior.temporal.layers):
            h, k, v = _ar_layer(layer, h, cos, sin, past_k[index], past_v[index], bias=attention_bias)
            keys.append(k)
            values.append(v)
        logits = self.model.ar_prior.token_head(self.model.ar_prior.out_norm(h[:, -1:, :]))[:, 0]
        return logits, torch.stack(keys), torch.stack(values)


class AcousticConditionGraph(nn.Module):
    def __init__(self, model: SoproModel) -> None:
        super().__init__()
        self.head = model.acoustic_head

    def forward(self, semantic_tokens: torch.Tensor, frame_to_token: torch.Tensor):
        head = self.head
        latents = head.semantic_latents(semantic_tokens, head.semantic_token_emb.weight.dtype)
        prelook = head.semantic_prelook(latents)
        repeated = prelook.index_select(2, frame_to_token)
        up = head.semantic_upsampler
        hidden = F.silu(up.in_proj(repeated))
        hidden = F.silu(up.mix(hidden))
        return head.mu_proj(repeated + up.out_proj(hidden))


def _acoustic_layer(block: nn.Module, h: torch.Tensor, emb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_k: torch.Tensor | None = None, past_v: torch.Tensor | None = None, bias: torch.Tensor | None = None):
    normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb)
    attn = block.attn
    b, n, _ = normed.shape
    q = apply_rotary(attn.to_q(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2), cos, sin)
    k = apply_rotary(attn.to_k(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2), cos, sin)
    v = attn.to_v(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2)
    present_k = k if past_k is None else torch.cat([past_k, k], dim=2)
    present_v = v if past_v is None else torch.cat([past_v, v], dim=2)
    y = _attention(q, present_k, present_v, bias)
    h = h + gate_msa[:, None, :] * attn.to_out(y.transpose(1, 2).contiguous().view(b, n, attn.heads * attn.dim_head))
    h_ff = block.ff_norm(h) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    return h + gate_mlp[:, None, :] * block.ff(h_ff), present_k, present_v


def _acoustic_input(head: nn.Module, x: torch.Tensor, mu: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    spk = head._speaker_embedding(cond_vec, x.dtype)
    h = head.input_embed(x, cond_mel, cond_mask, mu, spk)
    return h, spk


def _grid(steps: int, sway: float) -> List[float]:
    times = torch.linspace(0.0, 1.0, steps=steps + 1)
    if abs(sway) > 1.0e-8:
        times = times + sway * (torch.cos(0.5 * math.pi * times) - 1.0 + times)
    return [float(v) for v in times]


class AcousticOfflineGraph(nn.Module):
    def __init__(self, model: SoproModel, steps: int) -> None:
        super().__init__()
        self.head = model.acoustic_head
        self.steps = int(steps)

    def forward(self, x_init: torch.Tensor, mu: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        head, x = self.head, x_init
        mask = cond_mask.to(dtype=x.dtype).clamp(0.0, 1.0)
        prompt = cond_mel.to(dtype=x.dtype) * mask
        grid = _grid(self.steps, float(head.cfg.acoustic_sway_sampling_coef))
        for index in range(self.steps):
            t0, t1 = grid[index], grid[index + 1]
            emb = head._time_embedding(x.new_full((1,), t0), x.dtype)
            h, _ = _acoustic_input(head, x, mu, cond_vec, prompt, mask)
            for block in head.blocks:
                h, _, _ = _acoustic_layer(block, h, emb, cos, sin)
            velocity = head.out_proj(head.out_norm(h, emb)).transpose(1, 2).contiguous()
            x = x + (t1 - t0) * velocity
            x_prompt = (1.0 - (1.0 - float(head.cfg.acoustic_sigma_min)) * t1) * x_init + t1 * prompt
            x = mask * x_prompt + (1.0 - mask) * x
        return mask * prompt + (1.0 - mask) * x


class AcousticStreamPrefillGraph(nn.Module):
    def __init__(self, model: SoproModel, steps: int) -> None:
        super().__init__()
        self.head = model.acoustic_head
        self.steps = int(steps)

    def forward(self, x_init: torch.Tensor, mu: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, chunk_bias: torch.Tensor):
        head, x = self.head, x_init
        mask = cond_mask.to(dtype=x.dtype).clamp(0.0, 1.0)
        prompt = cond_mel.to(dtype=x.dtype) * mask
        grid = _grid(self.steps, float(head.cfg.acoustic_sway_sampling_coef))
        contexts, all_k, all_v = [], [], []
        for index in range(self.steps):
            t0, t1 = grid[index], grid[index + 1]
            emb = head._time_embedding(x.new_full((1,), t0), x.dtype)
            h, _ = _acoustic_input(head, x, mu, cond_vec, prompt, mask)
            layer_k, layer_v = [], []
            for block in head.blocks:
                h, k, v = _acoustic_layer(block, h, emb, cos, sin, bias=chunk_bias)
                # KV is stored fp16 with the cast placed per layer, so the fp32
                # copy of one layer is the only cache transient a run keeps.
                layer_k.append(k.half())
                layer_v.append(v.half())
            contexts.append(x[:, :, -POS_CONTEXT:])
            all_k.append(torch.stack(layer_k))
            all_v.append(torch.stack(layer_v))
            velocity = head.out_proj(head.out_norm(h, emb)).transpose(1, 2).contiguous()
            x = x + (t1 - t0) * velocity
            x_prompt = (1.0 - (1.0 - float(head.cfg.acoustic_sigma_min)) * t1) * x_init + t1 * prompt
            x = mask * x_prompt + (1.0 - mask) * x
        out = mask * prompt + (1.0 - mask) * x
        return out, torch.stack(contexts), torch.stack(all_k), torch.stack(all_v)


class AcousticStreamStepGraph(nn.Module):
    def __init__(self, model: SoproModel, steps: int) -> None:
        super().__init__()
        self.head = model.acoustic_head
        self.steps = int(steps)

    def forward(self, x_init: torch.Tensor, mu_window: torch.Tensor, cond_vec: torch.Tensor, cond_mel_window: torch.Tensor, cond_mask_window: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x_context: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        head, x = self.head, x_init
        new_count = x.shape[-1]
        mask_new = cond_mask_window[:, :, -new_count:].to(dtype=x.dtype).clamp(0.0, 1.0)
        prompt_new = cond_mel_window[:, :, -new_count:].to(dtype=x.dtype) * mask_new
        grid = _grid(self.steps, float(head.cfg.acoustic_sway_sampling_coef))
        contexts, all_k, all_v = [], [], []
        spk = head._speaker_embedding(cond_vec, x.dtype)
        ie = head.input_embed
        for index in range(self.steps):
            t0, t1 = grid[index], grid[index + 1]
            emb = head._time_embedding(x.new_full((1,), t0), x.dtype)
            x_win = torch.cat([x_context[index], x], dim=-1)
            n_win = x_win.shape[-1]
            h_all = ie.proj(torch.cat([
                x_win.transpose(1, 2),
                cond_mel_window.transpose(1, 2),
                mu_window.transpose(1, 2),
                spk[:, None, :].expand(-1, n_win, -1),
            ], dim=-1))
            h_all = h_all + ie.cond_mask_proj(cond_mask_window.transpose(1, 2).to(dtype=h_all.dtype))
            y = h_all.transpose(1, 2).contiguous()
            y = F.mish(ie.pos.conv1(y))
            y = F.mish(ie.pos.conv2(y))
            h = h_all[:, -new_count:] + y.transpose(1, 2).contiguous()
            layer_k, layer_v = [], []
            for layer_index, block in enumerate(head.blocks):
                h, k, v = _acoustic_layer(block, h, emb, cos, sin, past_k[index, layer_index].float(), past_v[index, layer_index].float())
                layer_k.append(k.half())
                layer_v.append(v.half())
            contexts.append(torch.cat([x_context[index], x], dim=-1)[:, :, -POS_CONTEXT:])
            all_k.append(torch.stack(layer_k))
            all_v.append(torch.stack(layer_v))
            velocity = head.out_proj(head.out_norm(h, emb)).transpose(1, 2).contiguous()
            x = x + (t1 - t0) * velocity
            x_prompt = (1.0 - (1.0 - float(head.cfg.acoustic_sigma_min)) * t1) * x_init + t1 * prompt_new
            x = mask_new * x_prompt + (1.0 - mask_new) * x
        out = mask_new * prompt_new + (1.0 - mask_new) * x
        return out, torch.stack(contexts), torch.stack(all_k), torch.stack(all_v)


class AcousticStreamPrefillOdeGraph(nn.Module):
    """One acoustic ODE solver step for the streaming prompt prefill.

    Keeping solver steps in separate ORT runs prevents WASM from retaining the
    first step's full attention workspace while evaluating the second step.
    The browser still executes the same solver grid and full-prefix attention.
    """

    def __init__(self, model: SoproModel, t0: float, t1: float) -> None:
        super().__init__()
        self.head = model.acoustic_head
        self.t0 = float(t0)
        self.t1 = float(t1)

    def forward(self, x: torch.Tensor, x_init: torch.Tensor, mu: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, chunk_bias: torch.Tensor):
        head = self.head
        mask = cond_mask.to(dtype=x.dtype).clamp(0.0, 1.0)
        prompt = cond_mel.to(dtype=x.dtype) * mask
        emb = head._time_embedding(x.new_full((1,), self.t0), x.dtype)
        h, _ = _acoustic_input(head, x, mu, cond_vec, prompt, mask)
        layer_k, layer_v = [], []
        for block in head.blocks:
            h, k, v = _acoustic_layer(block, h, emb, cos, sin, bias=chunk_bias)
            layer_k.append(k.half())
            layer_v.append(v.half())
        context = x[:, :, -POS_CONTEXT:]
        velocity = head.out_proj(head.out_norm(h, emb)).transpose(1, 2).contiguous()
        x = x + (self.t1 - self.t0) * velocity
        x_prompt = (1.0 - (1.0 - float(head.cfg.acoustic_sigma_min)) * self.t1) * x_init + self.t1 * prompt
        x = mask * x_prompt + (1.0 - mask) * x
        mel = mask * prompt + (1.0 - mask) * x
        return x, mel, context, torch.stack(layer_k), torch.stack(layer_v)


class AcousticStreamOdeGraph(nn.Module):
    """One cached acoustic ODE solver step for browser streaming."""

    def __init__(self, model: SoproModel, t0: float, t1: float) -> None:
        super().__init__()
        self.head = model.acoustic_head
        self.t0 = float(t0)
        self.t1 = float(t1)

    def forward(self, x: torch.Tensor, x_init: torch.Tensor, mu_window: torch.Tensor, cond_vec: torch.Tensor, cond_mel_window: torch.Tensor, cond_mask_window: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x_context: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        head = self.head
        new_count = x.shape[-1]
        mask_new = cond_mask_window[:, :, -new_count:].to(dtype=x.dtype).clamp(0.0, 1.0)
        prompt_new = cond_mel_window[:, :, -new_count:].to(dtype=x.dtype) * mask_new
        emb = head._time_embedding(x.new_full((1,), self.t0), x.dtype)
        spk = head._speaker_embedding(cond_vec, x.dtype)
        ie = head.input_embed
        x_win = torch.cat([x_context, x], dim=-1)
        n_win = x_win.shape[-1]
        h_all = ie.proj(torch.cat([
            x_win.transpose(1, 2),
            cond_mel_window.transpose(1, 2),
            mu_window.transpose(1, 2),
            spk[:, None, :].expand(-1, n_win, -1),
        ], dim=-1))
        h_all = h_all + ie.cond_mask_proj(cond_mask_window.transpose(1, 2).to(dtype=h_all.dtype))
        y = h_all.transpose(1, 2).contiguous()
        y = F.mish(ie.pos.conv1(y))
        y = F.mish(ie.pos.conv2(y))
        h = h_all[:, -new_count:] + y.transpose(1, 2).contiguous()
        layer_k, layer_v = [], []
        for layer_index, block in enumerate(head.blocks):
            h, k, v = _acoustic_layer(block, h, emb, cos, sin, past_k[layer_index].float(), past_v[layer_index].float())
            layer_k.append(k.half())
            layer_v.append(v.half())
        context = torch.cat([x_context, x], dim=-1)[:, :, -POS_CONTEXT:]
        velocity = head.out_proj(head.out_norm(h, emb)).transpose(1, 2).contiguous()
        x = x + (self.t1 - self.t0) * velocity
        x_prompt = (1.0 - (1.0 - float(head.cfg.acoustic_sigma_min)) * self.t1) * x_init + self.t1 * prompt_new
        x = mask_new * x_prompt + (1.0 - mask_new) * x
        mel = mask_new * prompt_new + (1.0 - mask_new) * x
        return x, mel, context, torch.stack(layer_k), torch.stack(layer_v)


def _flatten_vocoder_state(state: Mapping):
    conv = torch.stack([block["conv"] for block in state["blocks"]])
    pending = torch.stack([block["pending"] for block in state["blocks"]])
    return state["embed"], conv, pending


def _vocoder_state(embed: torch.Tensor, conv: torch.Tensor, pending: torch.Tensor) -> Dict:
    blocks = [{"conv": conv[index], "pending": pending[index]} for index in range(int(conv.shape[0]))]
    return {"embed": embed, "blocks": blocks}


def _vocoder_stream_conv(x: torch.Tensor, conv: nn.Conv1d, state: torch.Tensor, lookahead: int, flush: bool):
    context = int(conv.kernel_size[0]) - 1
    if flush and int(lookahead) > 0:
        x = F.pad(x, (0, int(lookahead)))
    available = torch.cat([state, x], dim=-1)
    y = F.conv1d(available, conv.weight, conv.bias, groups=int(conv.groups))
    return y, available[:, :, -context:]


def _vocoder_stream_backbone(vocoder: Vocoder, mel: torch.Tensor, state: Dict | None, flush: bool):
    backbone = vocoder.backbone
    if state is None:
        embed_context = int(backbone.embed.kernel_size[0]) - 1
        embed_state = mel.new_zeros((1, int(backbone.embed.in_channels), embed_context - int(backbone.lookahead_frames)))
        block_states = [None] * len(backbone.convnext)
    else:
        embed_state = state["embed"]
        block_states = state["blocks"]
    x, embed_out = _vocoder_stream_conv(mel, backbone.embed, embed_state, int(backbone.lookahead_frames), flush)
    x = backbone.norm(x.transpose(1, 2)).transpose(1, 2)
    outputs = []
    for index, block in enumerate(backbone.convnext):
        block_state = block_states[index]
        context = int(block.dwconv.kernel_size[0]) - 1
        if block_state is None:
            conv_state = x.new_zeros((1, int(block.dwconv.in_channels), context - int(block.lookahead)))
            pending = x.new_zeros((1, int(block.dwconv.in_channels), 0))
        else:
            conv_state, pending = block_state["conv"], block_state["pending"]
        residual_stream = torch.cat([pending, x], dim=-1)
        y, conv_out = _vocoder_stream_conv(x, block.dwconv, conv_state, int(block.lookahead), flush)
        residual = residual_stream[:, :, : y.shape[-1]]
        pending_out = residual_stream[:, :, y.shape[-1] :]
        x = residual + block._pointwise(y)
        outputs.append({"conv": conv_out, "pending": pending_out})
    return backbone.final_layer_norm(x.transpose(1, 2)), {"embed": embed_out, "blocks": outputs}


class VocoderStreamStartGraph(nn.Module):
    def __init__(self, vocoder: Vocoder) -> None:
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel: torch.Tensor):
        hidden, state = _vocoder_stream_backbone(self.vocoder, mel.float(), None, False)
        return (self.vocoder.head.out(hidden),) + _flatten_vocoder_state(state)


class VocoderStreamStepGraph(nn.Module):
    def __init__(self, vocoder: Vocoder) -> None:
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel: torch.Tensor, embed: torch.Tensor, conv: torch.Tensor, pending: torch.Tensor):
        hidden, state = _vocoder_stream_backbone(self.vocoder, mel.float(), _vocoder_state(embed, conv, pending), False)
        return (self.vocoder.head.out(hidden),) + _flatten_vocoder_state(state)


class VocoderStreamFlushGraph(nn.Module):
    def __init__(self, vocoder: Vocoder) -> None:
        super().__init__()
        self.vocoder = vocoder

    def forward(self, embed: torch.Tensor, conv: torch.Tensor, pending: torch.Tensor):
        mel = embed.new_zeros((1, int(self.vocoder.cfg.n_mels), 0))
        hidden, _ = _vocoder_stream_backbone(self.vocoder, mel, _vocoder_state(embed, conv, pending), True)
        return self.vocoder.head.out(hidden)


def _export(module: nn.Module, args: Tuple[torch.Tensor, ...], path: Path, input_names: Sequence[str], output_names: Sequence[str], dynamic_axes: Mapping[str, Mapping[int, str]]) -> None:
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dict(dynamic_axes),
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )


def _load_modules(root: Path):
    config = load_config(root)
    device = torch.device("cpu")
    model = SoproModel(config.model)
    model.load_state_dict(load_weights(root, MODEL_FILE, device))
    semantic = SemanticEncoder(config.semantic_encoder)
    semantic.load_state_dict(load_weights(root, SEMANTIC_ENCODER_FILE, device))
    speaker = SpeakerEncoder(config.speaker_encoder)
    speaker.load_state_dict(load_weights(root, SPEAKER_ENCODER_FILE, device))
    vocoder = Vocoder(config.vocoder)
    vocoder.load_state_dict(load_weights(root, VOCODER_FILE, device))
    for module in (model, semantic, speaker, vocoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return config, model, semantic, speaker, vocoder


def _export_fp32_graphs(root: Path, out: Path, steps: Sequence[int]) -> Tuple[SoproConfig, Dict[str, str]]:
    config, model, semantic, speaker, vocoder = _load_modules(root)
    cfg = config.model
    graphs: Dict[str, str] = {}

    def emit(name: str, module: nn.Module, args, inputs, outputs, axes):
        filename = f"{name}.onnx"
        _export(module, args, out / filename, inputs, outputs, axes)
        graphs[name] = filename

    semantic_mel = torch.randn(1, 80, 103)
    interp_left = torch.arange(20, dtype=torch.long).clamp_max(49)
    interp_right = (interp_left + 1).clamp_max(50)
    interp_weight = torch.zeros(20)
    speaker_mel = torch.randn(1, 80, 101)
    emit("reference", ReferenceGraph(model, semantic, speaker), (semantic_mel, interp_left, interp_right, interp_weight, speaker_mel),
         ("semantic_mel", "interp_left", "interp_right", "interp_weight", "speaker_mel"), ("cond_vec", "semantic_tokens"),
         {"semantic_mel": {2: "semantic_frames"}, "interp_left": {0: "semantic_tokens"}, "interp_right": {0: "semantic_tokens"}, "interp_weight": {0: "semantic_tokens"}, "speaker_mel": {2: "speaker_frames"}, "semantic_tokens": {1: "semantic_tokens"}})

    text_ids = torch.randint(0, int(cfg.text_vocab_size), (1, 12))
    style = torch.randint(0, int(cfg.semantic_vocab_size), (1, 20))
    prompt = torch.randint(0, int(cfg.semantic_vocab_size), (1, 16))
    prefix_len = int(cfg.style_prefix_tokens) + 12 + 16 + 1
    cos = torch.randn(prefix_len, int(cfg.ar_model_dim) // int(cfg.ar_heads))
    sin = torch.randn_like(cos)
    bias = torch.zeros(1, 1, prefix_len, prefix_len).masked_fill(~torch.ones(prefix_len, prefix_len, dtype=torch.bool).tril()[None, None], -1.0e4)
    emit("semantic_prefix", SemanticPrefixEmbeddingGraph(model), (text_ids, style, prompt),
         ("text_ids", "style_tokens", "prompt_tokens"), ("hidden", "bos_hidden"),
         {"text_ids": {1: "text_tokens"}, "style_tokens": {1: "style_tokens"}, "prompt_tokens": {1: "prompt_tokens"}, "hidden": {1: "hidden_tokens"}})
    past_k = torch.randn(int(cfg.ar_blocks), 1, int(cfg.ar_kv_heads), 1, int(cfg.ar_model_dim) // int(cfg.ar_heads))
    past_v = torch.randn_like(past_k)
    hidden = torch.randn(1, prefix_len - 1, int(cfg.ar_model_dim))
    core_bias = torch.zeros(1, 1, prefix_len, prefix_len + 1)
    emit("semantic_core", SemanticCoreGraph(model), (hidden, torch.ones(1, 1, dtype=torch.long), past_k, past_v, cos, sin, core_bias),
         ("hidden", "token", "past_k", "past_v", "cos", "sin", "attention_bias"), ("logits", "present_k", "present_v"),
         {"hidden": {1: "hidden_tokens"}, "past_k": {3: "past_tokens"}, "past_v": {3: "past_tokens"}, "cos": {0: "query_tokens"}, "sin": {0: "query_tokens"}, "attention_bias": {2: "query_tokens", 3: "present_tokens"}, "present_k": {3: "present_tokens"}, "present_v": {3: "present_tokens"}})
    sem_tokens = torch.randint(0, int(cfg.semantic_vocab_size), (1, 36))
    frame_map = torch.div(torch.arange(144) * 36, 144, rounding_mode="floor")
    emit("acoustic_condition", AcousticConditionGraph(model), (sem_tokens, frame_map), ("semantic_tokens", "frame_to_token"), ("mu",),
         {"semantic_tokens": {1: "semantic_tokens"}, "frame_to_token": {0: "mel_frames"}, "mu": {2: "mel_frames"}})

    frames = 128
    x0 = torch.randn(1, int(cfg.acoustic_mel_n_mels), frames)
    mu = torch.randn(1, int(cfg.acoustic_mu_dim), frames)
    cond = torch.randn(1, int(cfg.cond_hidden_dim))
    cond_mel = torch.randn_like(x0)
    cond_mask = torch.zeros(1, 1, frames)
    ac_cos = torch.randn(frames, int(cfg.acoustic_dit_dim_head))
    ac_sin = torch.randn_like(ac_cos)
    chunk = 64
    positions = torch.arange(frames)
    ends = ((torch.div(positions, chunk, rounding_mode="floor") + 1) * chunk).clamp_max(frames)
    chunk_bias = torch.zeros(1, 1, frames, frames).masked_fill(~(positions[None, :] < ends[:, None])[None, None], -1.0e4)
    for n_steps in steps:
        suffix = str(int(n_steps))
        grid = _grid(n_steps, float(model.acoustic_head.cfg.acoustic_sway_sampling_coef))
        emit(f"acoustic_offline_{suffix}", AcousticOfflineGraph(model, n_steps), (x0, mu, cond, cond_mel, cond_mask, ac_cos, ac_sin),
             ("x_init", "mu", "cond_vec", "cond_mel", "cond_mask", "cos", "sin"), ("mel",),
             {"x_init": {2: "mel_frames"}, "mu": {2: "mel_frames"}, "cond_mel": {2: "mel_frames"}, "cond_mask": {2: "mel_frames"}, "cos": {0: "mel_frames"}, "sin": {0: "mel_frames"}, "mel": {2: "mel_frames"}})
        emit(f"acoustic_stream_prefill_{suffix}", AcousticStreamPrefillGraph(model, n_steps), (x0, mu, cond, cond_mel, torch.ones_like(cond_mask), ac_cos, ac_sin, chunk_bias),
             ("x_init", "mu", "cond_vec", "cond_mel", "cond_mask", "cos", "sin", "chunk_bias"), ("mel", "x_context", "present_k", "present_v"),
             {"x_init": {2: "prompt_frames"}, "mu": {2: "prompt_frames"}, "cond_mel": {2: "prompt_frames"}, "cond_mask": {2: "prompt_frames"}, "cos": {0: "prompt_frames"}, "sin": {0: "prompt_frames"}, "chunk_bias": {2: "prompt_frames", 3: "prompt_frames"}, "mel": {2: "prompt_frames"}, "present_k": {4: "prompt_frames"}, "present_v": {4: "prompt_frames"}})
        for step in range(n_steps):
            emit(f"acoustic_stream_prefill_ode_{suffix}_{step}", AcousticStreamPrefillOdeGraph(model, grid[step], grid[step + 1]), (x0, x0, mu, cond, cond_mel, torch.ones_like(cond_mask), ac_cos, ac_sin, chunk_bias),
                 ("x", "x_init", "mu", "cond_vec", "cond_mel", "cond_mask", "cos", "sin", "chunk_bias"), ("x_out", "mel", "x_context_out", "present_k", "present_v"),
                 {"x": {2: "prompt_frames"}, "x_init": {2: "prompt_frames"}, "mu": {2: "prompt_frames"}, "cond_mel": {2: "prompt_frames"}, "cond_mask": {2: "prompt_frames"}, "cos": {0: "prompt_frames"}, "sin": {0: "prompt_frames"}, "chunk_bias": {2: "prompt_frames", 3: "prompt_frames"}, "x_out": {2: "prompt_frames"}, "mel": {2: "prompt_frames"}, "present_k": {3: "prompt_frames"}, "present_v": {3: "prompt_frames"}})
        past_frames, new_frames = 128, 64
        x_new = torch.randn(1, int(cfg.acoustic_mel_n_mels), new_frames)
        mu_win = torch.randn(1, int(cfg.acoustic_mu_dim), POS_CONTEXT + new_frames)
        mel_win = torch.randn(1, int(cfg.acoustic_mel_n_mels), POS_CONTEXT + new_frames)
        mask_win = torch.zeros(1, 1, POS_CONTEXT + new_frames)
        x_ctx = torch.randn(n_steps, 1, int(cfg.acoustic_mel_n_mels), POS_CONTEXT)
        k_cache = torch.randn(n_steps, int(cfg.acoustic_dit_depth), 1, int(cfg.acoustic_dit_heads), past_frames, int(cfg.acoustic_dit_dim_head)).half()
        v_cache = torch.randn_like(k_cache.float()).half()
        emit(f"acoustic_stream_step_{suffix}", AcousticStreamStepGraph(model, n_steps), (x_new, mu_win, cond, mel_win, mask_win, ac_cos[:new_frames], ac_sin[:new_frames], x_ctx, k_cache, v_cache),
             ("x_init", "mu_window", "cond_vec", "cond_mel_window", "cond_mask_window", "cos", "sin", "x_context", "past_k", "past_v"), ("mel", "x_context_out", "present_k", "present_v"),
             {"x_init": {2: "new_frames"}, "mu_window": {2: "window_frames"}, "cond_mel_window": {2: "window_frames"}, "cond_mask_window": {2: "window_frames"}, "cos": {0: "new_frames"}, "sin": {0: "new_frames"}, "past_k": {4: "past_frames"}, "past_v": {4: "past_frames"}, "mel": {2: "new_frames"}, "present_k": {4: "present_frames"}, "present_v": {4: "present_frames"}})
        for step in range(n_steps):
            emit(f"acoustic_stream_ode_{suffix}_{step}", AcousticStreamOdeGraph(model, grid[step], grid[step + 1]), (x_new, x_new, mu_win, cond, mel_win, mask_win, ac_cos[:new_frames], ac_sin[:new_frames], x_ctx[step], k_cache[step], v_cache[step]),
                 ("x", "x_init", "mu_window", "cond_vec", "cond_mel_window", "cond_mask_window", "cos", "sin", "x_context", "past_k", "past_v"), ("x_out", "mel", "x_context_out", "present_k", "present_v"),
                 {"x": {2: "new_frames"}, "x_init": {2: "new_frames"}, "mu_window": {2: "window_frames"}, "cond_mel_window": {2: "window_frames"}, "cond_mask_window": {2: "window_frames"}, "cos": {0: "new_frames"}, "sin": {0: "new_frames"}, "past_k": {3: "past_frames"}, "past_v": {3: "past_frames"}, "x_out": {2: "new_frames"}, "mel": {2: "new_frames"}, "present_k": {3: "present_frames"}, "present_v": {3: "present_frames"}})

    stream_mel = torch.randn(1, int(config.vocoder.n_mels), 64)
    stream_outputs = ("istft_features", "embed_state_out", "conv_state_out", "pending_out")
    stream_axes = {"mel": {2: "mel_frames"}, "istft_features": {1: "output_frames"}}
    emit("vocoder_stream_start", VocoderStreamStartGraph(vocoder), (stream_mel,), ("mel",), stream_outputs, stream_axes)
    with torch.inference_mode():
        _, state = vocoder.backbone.forward_stream(stream_mel, None, False)
        embed, conv, pending = _flatten_vocoder_state(state)
    emit("vocoder_stream_step", VocoderStreamStepGraph(vocoder), (stream_mel, embed, conv, pending), ("mel", "embed_state", "conv_state", "pending"), stream_outputs, stream_axes)
    emit("vocoder_stream_flush", VocoderStreamFlushGraph(vocoder), (embed, conv, pending), ("embed_state", "conv_state", "pending"), ("istft_features",), {})
    return config, graphs


def _tokenizer_json(model_path: Path, output: Path) -> None:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    pieces = [processor.id_to_piece(i) for i in range(processor.get_piece_size())]
    data = {
        "format": "sentencepiece-unigram-v1",
        "pieces": pieces,
        "scores": [processor.get_score(i) for i in range(processor.get_piece_size())],
        "types": ["byte" if processor.is_byte(i) else "control" if processor.is_control(i) else "unused" if processor.is_unused(i) else "unknown" if processor.is_unknown(i) else "normal" for i in range(processor.get_piece_size())],
        "bosId": int(processor.bos_id()),
        "eosId": int(processor.eos_id()),
        "unkId": int(processor.unk_id()),
        "maxLength": 512,
    }
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_for_graph(name: str) -> str:
    if name == "reference":
        return "reference"
    if name.startswith("semantic_"):
        return "semantic"
    if name == "acoustic_condition":
        # The condition graph uses a few MB of the acoustic weights. Giving it
        # a dedicated shard keeps session creation from staging the full DiT
        # shard through browser memory for a 10 KB graph.
        return "condition"
    if name.startswith("acoustic_"):
        return "acoustic"
    return "vocoder"


def _merge_vocoder_stream_graphs(profile_dir: Path, graphs: Dict[str, str]) -> None:
    """Put start, step, and flush behind one runtime-selected ONNX session.

    All three exported graphs contain the same streaming-vocoder parameters.
    ORT otherwise materializes those parameters independently for every
    session and, on WebGPU, uploads them to the GPU three times.
    The branch bodies are copied verbatim after quantization; only their
    identical initializers are hoisted and deduplicated in the parent graph.
    """
    import copy

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source_names = ("vocoder_stream_start", "vocoder_stream_step", "vocoder_stream_flush")
    if not all(name in graphs for name in source_names):
        return

    models = {name: onnx.load_model(str(profile_dir / graphs[name]), load_external_data=True) for name in source_names}
    step = models["vocoder_stream_step"]
    output_names = [value.name for value in step.graph.output]

    # Use generated parent-scope names so unrelated PyTorch node numbering in
    # the three traces cannot make distinct tensors alias each other.
    canonical_by_value: Dict[Tuple[int, Tuple[int, ...], bytes], str] = {}
    parent_initializers = []

    def branch(name: str):
        graph = copy.deepcopy(models[name].graph)
        replacements = {}
        for initializer in graph.initializer:
            array = numpy_helper.to_array(initializer)
            key = (int(initializer.data_type), tuple(int(v) for v in initializer.dims), array.tobytes(order="C"))
            canonical = canonical_by_value.get(key)
            if canonical is None:
                canonical = f"vocoder_stream_initializer_{len(parent_initializers)}"
                canonical_by_value[key] = canonical
                parent_initializers.append(numpy_helper.from_array(array, name=canonical))
            replacements[initializer.name] = canonical
        for node in graph.node:
            for index, value in enumerate(node.input):
                if value in replacements:
                    node.input[index] = replacements[value]
        del graph.initializer[:]
        del graph.input[:]
        return graph

    start_branch = branch("vocoder_stream_start")
    step_branch = branch("vocoder_stream_step")
    flush_branch = branch("vocoder_stream_flush")

    # If requires both branches to expose the same contract. Flush does not
    # mutate state, so pass its input state through for the three outputs the JS
    # caller intentionally does not request.
    step_inputs = {value.name: value for value in step.graph.input}
    for input_name, output_name in zip(("embed_state", "conv_state", "pending"), output_names[1:]):
        flush_branch.node.append(helper.make_node("Identity", [input_name], [output_name], name=f"flush_{output_name}"))
        value = copy.deepcopy(step_inputs[input_name])
        value.name = output_name
        flush_branch.output.append(value)

    nonstart_outputs = [copy.deepcopy(value) for value in step.graph.output]
    nonstart_names = [f"nonstart_{name}" for name in output_names]
    for value, renamed in zip(nonstart_outputs, nonstart_names):
        value.name = renamed
    nonstart = helper.make_graph(
        [helper.make_node("If", ["is_flush"], nonstart_names, name="vocoder_flush_or_step", then_branch=flush_branch, else_branch=step_branch)],
        "vocoder_stream_nonstart",
        [],
        nonstart_outputs,
    )

    inputs = [helper.make_tensor_value_info("is_start", TensorProto.BOOL, []), helper.make_tensor_value_info("is_flush", TensorProto.BOOL, [])]
    inputs.extend(copy.deepcopy(value) for value in step.graph.input)
    graph = helper.make_graph(
        [helper.make_node("If", ["is_start"], output_names, name="vocoder_start_or_continue", then_branch=start_branch, else_branch=nonstart)],
        "vocoder_stream",
        inputs,
        [copy.deepcopy(value) for value in step.graph.output],
        initializer=parent_initializers,
    )
    merged = helper.make_model(graph, producer_name="sopro-web-export", opset_imports=copy.deepcopy(step.opset_import))
    merged.ir_version = step.ir_version
    merged_path = profile_dir / "vocoder.onnx"
    onnx.checker.check_model(merged)
    onnx.save_model(merged, str(merged_path))

    for name in source_names:
        (profile_dir / graphs.pop(name)).unlink()
    graphs["vocoder"] = merged_path.name


def _pack_external(profile_dir: Path, graphs: Mapping[str, str]):
    import onnx
    from onnx import numpy_helper
    from onnx.external_data_helper import set_external_data

    by_group: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for name, filename in graphs.items():
        by_group[_group_for_graph(name)].append((name, filename))
    graph_meta, shard_meta = {}, {}
    for group, members in by_group.items():
        shard_name = f"weights-{group}.bin"
        shard_path = profile_dir / shard_name
        offsets: Dict[str, Tuple[int, int]] = {}
        with shard_path.open("wb") as shard:
            for _, filename in members:
                model = onnx.load_model(str(profile_dir / filename), load_external_data=True)
                for initializer in model.graph.initializer:
                    array = numpy_helper.to_array(initializer)
                    raw = array.tobytes(order="C")
                    if len(raw) < 1024:
                        continue
                    key = hashlib.sha256(str(initializer.data_type).encode() + str(tuple(initializer.dims)).encode() + raw).hexdigest()
                    if key not in offsets:
                        padding = (-shard.tell()) % 64
                        if padding:
                            shard.write(b"\0" * padding)
                        offsets[key] = (shard.tell(), len(raw))
                        shard.write(raw)
            shard.flush()
        for name, filename in members:
            path = profile_dir / filename
            model = onnx.load_model(str(path), load_external_data=True)
            for initializer in model.graph.initializer:
                array = numpy_helper.to_array(initializer)
                raw = array.tobytes(order="C")
                if len(raw) < 1024:
                    # Small tensors stay inline: ONNX shape inference cannot
                    # read values (Reshape shapes, zero points) from external
                    # data.
                    initializer.CopyFrom(numpy_helper.from_array(array, name=initializer.name))
                    continue
                key = hashlib.sha256(str(initializer.data_type).encode() + str(tuple(initializer.dims)).encode() + raw).hexdigest()
                offset, length = offsets[key]
                initializer.CopyFrom(numpy_helper.from_array(array, name=initializer.name))
                set_external_data(initializer, location=shard_name, offset=offset, length=length)
                initializer.ClearField("raw_data")
                initializer.ClearField("float_data")
                initializer.ClearField("int32_data")
                initializer.ClearField("int64_data")
                initializer.ClearField("double_data")
                initializer.ClearField("uint64_data")
            onnx.save_model(model, str(path))
            graph_meta[name] = {"file": filename, "shard": shard_name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        shard_meta[group] = {"file": shard_name, "bytes": shard_path.stat().st_size, "sha256": _sha256(shard_path)}
    return graph_meta, shard_meta


def _make_profile(source: Path, output: Path, profile: str, graphs: Mapping[str, str]):
    import onnx

    def quantize_semantic(input_path: Path, output_path: Path) -> None:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

    def quantize_acoustic(input_path: Path, output_path: Path, op_types: Sequence[str] = ("MatMul",)) -> None:
        # quantize_dynamic mishandles transB Gemm (emits a MatMulInteger with
        # untransposed weights), so rewrite every initializer-weighted Gemm as
        # MatMul(+Add) with pre-transposed weights, then quantize MatMuls only.
        import numpy as np
        from onnx import helper, numpy_helper
        from onnxruntime.quantization import QuantType, quantize_dynamic

        model = onnx.load_model(str(input_path), load_external_data=True)
        weights = {item.name: item for item in model.graph.initializer}
        rewritten = []
        for node in model.graph.node:
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            eligible = (
                node.op_type == "Gemm"
                and node.input[1] in weights
                and attributes.get("alpha", 1.0) == 1.0
                and attributes.get("beta", 1.0) == 1.0
                and attributes.get("transA", 0) == 0
            )
            if not eligible:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weights[node.input[1]])
            if attributes.get("transB", 0) == 1:
                array = array.T
            transposed = node.input[1] + "_t"
            model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array), name=transposed))
            bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
            product = node.output[0] + "_mm" if bias else node.output[0]
            rewritten.append(helper.make_node("MatMul", [node.input[0], transposed], [product], name=node.name + "_MatMul"))
            if bias:
                rewritten.append(helper.make_node("Add", [product, bias], [node.output[0]], name=node.name + "_Add"))
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
        used = {name for node in model.graph.node for name in node.input}
        kept = [item for item in model.graph.initializer if item.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        staged = output_path.with_suffix(".pre.onnx")
        onnx.save_model(model, str(staged))
        quantize_dynamic(
            str(staged),
            str(output_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
            op_types_to_quantize=list(op_types),
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        staged.unlink()

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if profile == "webgpu-fp16":
        from onnxconverter_common.float16 import convert_float_to_float16

        for name, filename in graphs.items():
            if name == "reference":
                shutil.copy2(source / filename, output / filename)
                continue
            if name.startswith("semantic_"):
                quantize_semantic(source / filename, output / filename)
                continue
            if name.startswith("vocoder_"):
                shutil.copy2(source / filename, output / filename)
                continue
            model = onnx.load_model(str(source / filename))
            model = convert_float_to_float16(model, keep_io_types=False, disable_shape_infer=False, check_fp16_ready=False)
            # PyTorch emits explicit `Cast(..., to=float32)` nodes for numerically
            # sensitive eager-mode paths. The generic converter changes their
            # value annotations to fp16 but leaves the `to` attribute unchanged,
            # which ORT correctly rejects. Acoustic and vocoder graphs are
            # deliberately all-fp16, so keep their contracts consistent as well.
            for node in model.graph.node:
                if node.op_type == "Cast":
                    for attribute in node.attribute:
                        if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                            attribute.i = onnx.TensorProto.FLOAT16
            onnx.save_model(model, str(output / filename))
        dtype, provider = "float16", "webgpu"
    elif profile in ("wasm-uint8", "webgpu-fp32"):
        for name, filename in graphs.items():
            # The semantic transformer is uint8 everywhere. The wasm fallback
            # also carries int8 acoustic graphs (solver measured transparent:
            # mel MAE ~2e-3) so low-memory phones stay under tab limits; the
            # reference encoder and vocoder remain fp32.
            if profile == "wasm-uint8" and name == "acoustic_condition":
                # The conditioner is conv stacks plus an embedding table; its
                # pointwise convs are matmuls in disguise and the table is
                # int8-tolerant, so both are quantized here.
                quantize_acoustic(source / filename, output / filename, ("Conv", "Gather", "MatMul"))
            elif profile == "wasm-uint8" and name.startswith("acoustic_"):
                quantize_acoustic(source / filename, output / filename)
            elif profile == "wasm-uint8" and name.startswith("vocoder_"):
                # Mobile-only int8 vocoder: 92 -> 24 MB resident and download,
                # measured UTMOS -0.06..-0.08 against fp32. Desktop profiles
                # keep the fp32 vocoder.
                quantize_acoustic(source / filename, output / filename, ("Conv", "MatMul"))
            elif name == "reference" or name.startswith(("acoustic_", "vocoder_")):
                shutil.copy2(source / filename, output / filename)
            else:
                quantize_semantic(source / filename, output / filename)
        dtype, provider = ("float32", "wasm") if profile == "wasm-uint8" else ("float32", "webgpu")
    else:
        raise ValueError(f"unknown profile {profile!r}")
    profile_graphs = dict(graphs)
    _merge_vocoder_stream_graphs(output, profile_graphs)
    graph_meta, shards = _pack_external(output, profile_graphs)
    for name, item in graph_meta.items():
        if profile in ("webgpu-fp16", "webgpu-fp32") and (name == "reference" or name.startswith("semantic_")):
            item.update(dtype="float32", provider="wasm")
        elif profile == "webgpu-fp16" and name == "vocoder":
            item.update(dtype="float32", provider="webgpu")
        else:
            item.update(dtype=dtype, provider=provider)
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for item in graph_meta.values():
        graph_path = str(output / item["file"])
        onnx.checker.check_model(graph_path)
        # The ONNX checker does not catch every inferred dtype conflict. Session
        # construction does, and also verifies that external-data offsets mount.
        # Native CPU ORT rewrites some valid fp16 reductions back to fp32, so the
        # WebGPU profile is exercised in the browser instead.
        if item["provider"] != "webgpu":
            session = ort.InferenceSession(graph_path, sess_options=session_options, providers=["CPUExecutionProvider"])
            del session
    return {"dtype": dtype, "provider": provider, "graphs": graph_meta, "shards": shards}


def _deduplicate_profiles(output: Path, profiles: Mapping[str, dict]) -> None:
    """Move byte-identical cross-profile shard units into one shared directory.

    A unit is a shard plus every graph referencing it. Moving units whole keeps
    native ONNX external-data resolution (shard next to graph) valid in every
    directory, not just through the manifest.
    """
    units = defaultdict(list)
    for profile_name, profile in profiles.items():
        graphs_by_shard = defaultdict(list)
        for item in profile["graphs"].values():
            graphs_by_shard[item["shard"]].append(item)
        for shard_item in profile["shards"].values():
            members = [shard_item] + sorted(graphs_by_shard[shard_item["file"]], key=lambda item: item["file"])
            key = tuple((item["file"], item["sha256"]) for item in members)
            units[key].append((profile_name, members))
    common = output / "common"
    for entries in units.values():
        if len(entries) < 2:
            continue
        common.mkdir(parents=True, exist_ok=True)
        source_profile, source_items = entries[0]
        for item in source_items:
            shutil.move(output / source_profile / item["file"], common / item["file"])
        for profile_name, items in entries:
            for item in items:
                path = output / profile_name / item["file"]
                if path.exists():
                    path.unlink()
                item["url"] = f"../common/{item['file']}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True, help="Shipping safetensors/tokenizer directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory served to the browser")
    parser.add_argument("--model-name", default="sopro-v2-turbo", help="Public model id embedded in the manifest")
    parser.add_argument("--profiles", default="webgpu-fp16,wasm-uint8", help="Comma-separated: webgpu-fp16, wasm-uint8; webgpu-fp32 is optional")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root, output = args.artifacts.resolve(), args.output.resolve()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    steps = [ACOUSTIC_STEPS]
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sopro-onnx-", dir=str(output.parent)) as tmp:
        base = Path(tmp)
        config, graphs = _export_fp32_graphs(root, base, steps)
        profile_meta = {}
        for profile in profiles:
            profile_meta[profile] = _make_profile(base, output / profile, profile, graphs)
        _deduplicate_profiles(output, profile_meta)
    _tokenizer_json(root / TOKENIZER_FILE, output / "tokenizer.json")
    tokenizer_sha = _sha256(output / "tokenizer.json")
    revision = hashlib.sha256(
        json.dumps({"tokenizer": tokenizer_sha, "profiles": profile_meta}, sort_keys=True).encode()
    ).hexdigest()[:16]
    manifest = {
        "format": 1,
        "model": args.model_name,
        "revision": revision,
        "opset": OPSET,
        "sampleRate": int(config.sample_rate),
        "hopRatio": int(config.semantic_encoder.token_samples_24k) // int(config.model.acoustic_mel_hop_length),
        "tokenSamples": int(config.semantic_encoder.token_samples_24k),
        "positionContext": POS_CONTEXT,
        "steps": ACOUSTIC_STEPS,
        "acousticGrid": _grid(ACOUSTIC_STEPS, float(config.model.acoustic_sway_sampling_coef)),
        "config": config.to_dict(),
        "tokenizer": "tokenizer.json",
        "profiles": profile_meta,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
