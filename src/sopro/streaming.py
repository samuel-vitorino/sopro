from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

from .codec.mimi import MimiDecodeState, MimiStreamDecoder
from .model import SoproTTS, SoproTTSModel


@dataclass
class StreamConfig:
    chunk_frames: int = 16
    nar_context_frames: Optional[int] = None
    cond_chunk_size: int = 32


class SoproTTSStreamer:
    def __init__(self, tts: SoproTTS, cfg: Optional[StreamConfig] = None):
        self.tts = tts
        self.cfg = cfg or StreamConfig()
        self.mimi_stream = MimiStreamDecoder(tts.codec)

    @torch.inference_mode()
    def stream(
        self,
        text: str,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        max_frames: int = 400,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        use_prefix: bool = True,
        prefix_sec_fixed: Optional[float] = None,
        style_strength: Optional[float] = None,
        ref_seconds: Optional[float] = None,
        chunk_frames: Optional[int] = None,
        nar_context_frames: Optional[int] = None,
        cond_chunk_size: Optional[int] = None,
        use_stop_head: Optional[bool] = None,
        stop_patience: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        min_gen_frames: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        model: SoproTTSModel = self.tts.model
        device = self.tts.device

        text_ids = self.tts.encode_text(text)
        ref = self.tts.encode_reference(
            ref_audio_path=ref_audio_path,
            ref_tokens_tq=ref_tokens_tq,
            ref_seconds=ref_seconds,
        )

        prep = model.prepare_conditioning_lazy(
            text_ids,
            ref,
            max_frames=max_frames,
            device=device,
            style_strength=float(
                style_strength
                if style_strength is not None
                else self.tts.cfg.style_strength
            ),
        )

        cf = int(chunk_frames if chunk_frames is not None else self.cfg.chunk_frames)
        cond_cs = int(
            cond_chunk_size if cond_chunk_size is not None else self.cfg.cond_chunk_size
        )

        nar_ctx = (
            nar_context_frames
            if nar_context_frames is not None
            else self.cfg.nar_context_frames
        )
        if nar_ctx is None:
            nar_ctx = int(model.rf_nar())
        nar_ctx = int(nar_ctx)

        hist_A: List[int] = []

        frames_emitted = 0

        mimi_state = MimiDecodeState()

        def refine_and_emit(end: int) -> Optional[torch.Tensor]:
            nonlocal frames_emitted, mimi_state

            new_start = frames_emitted
            if end <= new_start:
                return None

            win_start = max(0, new_start - nar_ctx)
            win_end = end

            cond_win = prep["cond_all"][:, win_start:win_end, :]
            tokens_A_win = torch.as_tensor(
                hist_A[win_start:win_end], device=device, dtype=torch.long
            ).unsqueeze(0)

            tokens_win_tq = model.nar_refine(cond_win, tokens_A_win).squeeze(0)

            tail_i = new_start - win_start
            emit_tokens = tokens_win_tq[tail_i:, :]

            wav_chunk, mimi_state = self.mimi_stream.decode_step(
                emit_tokens, mimi_state
            )
            frames_emitted = end
            return wav_chunk if wav_chunk.numel() > 0 else None

        for _t, rvq1_id, _p_stop in model.ar_stream(
            prep,
            max_frames=max_frames,
            top_p=top_p,
            temperature=temperature,
            anti_loop=anti_loop,
            use_prefix=use_prefix,
            prefix_sec_fixed=prefix_sec_fixed,
            cond_chunk_size=cond_cs,
            use_stop_head=use_stop_head,
            stop_patience=stop_patience,
            stop_threshold=stop_threshold,
            min_gen_frames=min_gen_frames,
        ):
            hist_A.append(int(rvq1_id))
            T = len(hist_A)

            is_boundary = (T % cf) == 0
            if not is_boundary and T < max_frames:
                continue

            wav = refine_and_emit(T)
            if wav is not None:
                yield wav

        T_final = len(hist_A)
        if frames_emitted < T_final:
            wav = refine_and_emit(T_final)
            if wav is not None:
                yield wav

@torch.inference_mode()
def stream(
    tts: SoproTTS,
    text: str,
    *,
    ref_audio_path: Optional[str] = None,
    ref_tokens_tq: Optional[torch.Tensor] = None,
    chunk_frames: int = 6,
    **kwargs,
) -> Iterator[torch.Tensor]:
    streamer = SoproTTSStreamer(tts, StreamConfig(chunk_frames=chunk_frames))
    return streamer.stream(
        text,
        ref_audio_path=ref_audio_path,
        ref_tokens_tq=ref_tokens_tq,
        chunk_frames=chunk_frames,
        **kwargs,
    )
