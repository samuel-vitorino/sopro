from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from sopro.nn.acoustic import ChunkedSolveState
from sopro.nn.model import SoproModel
from sopro.vocoder import Vocoder


@dataclass
class PromptState:
    steps: int
    chunk_frames: int
    frames: int
    noise: torch.Tensor
    state: ChunkedSolveState

    def to(self, device: torch.device) -> "PromptState":
        return PromptState(self.steps, self.chunk_frames, self.frames, self.noise.to(device), self.state.to(device))


@torch.inference_mode()
def build_prompt_state(model: SoproModel, ref_semantic_tokens: torch.Tensor, ref_mel: torch.Tensor, cond_vec: torch.Tensor, steps: int, chunk_frames: int, hop_ratio: int) -> PromptState:
    head = model.acoustic_head
    aligned = int(ref_mel.shape[-1]) // int(hop_ratio)
    p_mel = aligned * int(hop_ratio)
    final_tokens = aligned - int(head.semantic_prelook.lookahead)
    frames = (min(p_mel, final_tokens * int(hop_ratio)) // int(chunk_frames)) * int(chunk_frames)
    state = head.new_chunked_state(int(steps), max(1, frames))
    noise = torch.randn((1, int(ref_mel.shape[1]), p_mel), device=cond_vec.device, dtype=cond_vec.dtype)
    if frames > 0:
        cond_mask = ref_mel.new_ones((1, 1, p_mel))
        head.solve_chunked(state, noise, ref_semantic_tokens[:, :aligned], cond_vec, ref_mel[:, :, :p_mel], cond_mask, int(chunk_frames), frames)
    return PromptState(int(steps), int(chunk_frames), frames, noise, state)


class StreamSession:
    def __init__(
        self,
        model: SoproModel,
        vocoder: Vocoder,
        ref_semantic_tokens: torch.Tensor,
        ref_mel: torch.Tensor,
        cond_vec: torch.Tensor,
        mel_mean: torch.Tensor,
        mel_std: torch.Tensor,
        chunk_frames: int,
        hop_ratio: int,
        steps: int,
        warmup_frames: int,
        max_frames: int,
        prompt_state: Optional[PromptState] = None,
    ) -> None:
        self.model = model
        self.vocoder = vocoder
        self.cond_vec = cond_vec
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.chunk_frames = int(chunk_frames)
        self.hop_ratio = int(hop_ratio)
        self.steps = int(steps)
        aligned = int(ref_mel.shape[-1]) // self.hop_ratio
        self.ref_sem = ref_semantic_tokens[:, :aligned]
        self.ref_mel = ref_mel[:, :, : aligned * self.hop_ratio]
        self.p_sem = aligned
        self.p_mel = aligned * self.hop_ratio
        self.n_mels = int(ref_mel.shape[1])
        if prompt_state is not None and prompt_state.frames > 0:
            self.state = prompt_state.state.expanded(int(max_frames))
            self.x0 = prompt_state.noise
        else:
            self.state = model.acoustic_head.new_chunked_state(self.steps, int(max_frames))
            self.x0 = cond_vec.new_zeros((1, self.n_mels, 0))
        self.finality_margin = int(model.acoustic_head.semantic_prelook.lookahead) * self.hop_ratio
        self.tokens = ref_semantic_tokens.new_zeros((1, 0))
        self.kept = 0
        self.vocoder_state: Optional[Dict[str, Any]] = None
        self.skip_samples = int(warmup_frames) * int(vocoder.hop_length)
        if warmup_frames > 0:
            self._feed(self.ref_mel[:, :, self.p_mel - int(warmup_frames) :], flush=False)

    def _noise(self, t_max: int) -> torch.Tensor:
        have = int(self.x0.shape[-1])
        if have < t_max:
            self.x0 = torch.cat([self.x0, torch.randn((1, self.n_mels, t_max - have), device=self.cond_vec.device, dtype=self.cond_vec.dtype)], dim=-1)
        return self.x0[:, :, :t_max]

    def _feed(self, mel_model_space: torch.Tensor, flush: bool) -> torch.Tensor:
        mel = mel_model_space.float() * self.mel_std + self.mel_mean
        audio, self.vocoder_state = self.vocoder.decode_stream(mel, self.vocoder_state, flush=flush)
        audio = audio[0]
        if self.skip_samples > 0:
            drop = min(self.skip_samples, int(audio.shape[-1]))
            self.skip_samples -= drop
            audio = audio[drop:]
        return audio

    def _render(self, keep_end: int) -> torch.Tensor:
        n_tok = int(self.tokens.shape[-1])
        full_sem = torch.cat([self.ref_sem, self.tokens], dim=1)
        t_max = self.p_mel + n_tok * self.hop_ratio
        cond_mel = self.ref_mel.new_zeros((1, self.n_mels, t_max))
        cond_mask = self.ref_mel.new_zeros((1, 1, t_max))
        cond_mel[:, :, : self.p_mel] = self.ref_mel
        cond_mask[:, :, : self.p_mel] = 1.0
        cached = int(self.state.cached)
        out = self.model.acoustic_head.solve_chunked(self.state, self._noise(t_max), full_sem, self.cond_vec, cond_mel, cond_mask, self.chunk_frames, self.p_mel + keep_end)
        self.kept = int(keep_end)
        return out[:, :, max(0, self.p_mel - cached) :]

    def push(self, sem_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        self.tokens = torch.cat([self.tokens, sem_chunk], dim=1)
        covered = int(self.tokens.shape[-1]) * self.hop_ratio
        keep_end = ((self.p_mel + covered - self.finality_margin) // self.chunk_frames) * self.chunk_frames - self.p_mel
        if keep_end <= self.kept:
            return None
        return self._feed(self._render(keep_end), flush=False)

    def finish(self) -> torch.Tensor:
        covered = int(self.tokens.shape[-1]) * self.hop_ratio
        new = self._render(covered) if covered > self.kept else self.ref_mel.new_zeros((1, self.n_mels, 0))
        return self._feed(new, flush=True)
