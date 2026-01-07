from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from ..audio import center_crop_audio, load_audio_file, resample, trim_silence_energy
from ..constants import DEFAULT_MIMI_ID, TARGET_SR

try:
    from transformers import MimiConfig, MimiModel
except Exception:
    MimiModel = None
    MimiConfig = None


class MimiCodec:
    def __init__(
        self, num_quantizers: int, device: str = "cuda", model_id: str = DEFAULT_MIMI_ID
    ):
        if MimiModel is None or MimiConfig is None:
            raise RuntimeError(
                "MimiModel missing. Install a recent transformers version."
            )

        self.device = torch.device(device)
        cfg = MimiConfig.from_pretrained(model_id, num_quantizers=int(num_quantizers))
        self.model = (
            MimiModel.from_pretrained(model_id, config=cfg).to(self.device).eval()
        )

    @property
    def codebook_size(self) -> int:
        return int(getattr(self.model.config, "codebook_size", 2048))

    @property
    def num_quantizers(self) -> int:
        return int(getattr(self.model.config, "num_quantizers", 32))

    @torch.no_grad()
    def encode_file(
        self, wav_path: str, *, crop_seconds: Optional[float] = None
    ) -> torch.Tensor:
        wav, sr = load_audio_file(wav_path)

        wav = trim_silence_energy(wav, sr)

        cfg = self.model.config
        sr_target = int(getattr(cfg, "sampling_rate", TARGET_SR))
        wav = resample(wav, sr, sr_target, device=str(self.device))

        if crop_seconds is not None and crop_seconds > 0:
            fps = float(getattr(cfg, "frame_rate", 12.5))
            hop = int(round(sr_target / fps))
            win_frames = max(1, int(round(crop_seconds * fps)))
            win_samples = win_frames * hop
            wav = center_crop_audio(wav, win_samples)

        wav = wav.unsqueeze(0)
        out = self.model.encode(wav, return_dict=True)
        codes_bqt = out.audio_codes
        return codes_bqt[0].permute(1, 0).contiguous()

    @torch.no_grad()
    def decode_full(self, codes_tq: torch.Tensor) -> torch.Tensor:
        audio_codes = codes_tq.to(self.device).permute(1, 0).unsqueeze(0).contiguous()
        out = self.model.decode(audio_codes=audio_codes, return_dict=True)
        wav = out.audio_values
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav


@dataclass
class MimiDecodeState:
    decoder_past_key_values: Optional[object] = None
    frames_seen: int = 0
    samples_emitted: int = 0
    tail_codes_tq: Optional[torch.Tensor] = None


class MimiStreamDecoder:
    def __init__(self, codec: MimiCodec):
        self.codec = codec
        self.codec.model.config.use_cache = True

    def drop_cache_tail(self, pkv: Any, n: int):
        if pkv is None or n <= 0:
            return pkv

        if hasattr(pkv, "to_legacy_cache") and hasattr(type(pkv), "from_legacy_cache"):
            legacy = pkv.to_legacy_cache()

            trimmed = tuple(
                (
                    k[..., : max(0, k.shape[-2] - n), :].contiguous(),
                    v[..., : max(0, v.shape[-2] - n), :].contiguous(),
                )
                for (k, v) in legacy
            )
            return type(pkv).from_legacy_cache(trimmed)

        if isinstance(pkv, tuple):
            return tuple(
                (
                    k[..., : max(0, k.shape[-2] - n), :].contiguous(),
                    v[..., : max(0, v.shape[-2] - n), :].contiguous(),
                )
                for (k, v) in pkv
            )

        return pkv

    @torch.inference_mode()
    def decode_step(
        self,
        codes_chunk_tq: torch.Tensor,
        state: Optional[MimiDecodeState] = None,
        *,
        overlap_frames: int = 2,
    ) -> Tuple[torch.Tensor, MimiDecodeState]:
        if state is None:
            state = MimiDecodeState()

        cfg = self.codec.model.config
        sr = int(getattr(cfg, "sampling_rate", 24000))
        fps = float(getattr(cfg, "frame_rate", 12.5))
        hop = int(round(sr / fps))

        n_new = int(codes_chunk_tq.size(0))
        if n_new == 0:
            return torch.zeros(1, 0, device=self.codec.device), state

        tail = state.tail_codes_tq
        ov = 0
        if overlap_frames > 0 and tail is not None and tail.numel() > 0:
            ov = min(int(overlap_frames), int(tail.size(0)))
            tail = tail[-ov:]
            codes_in_tq = torch.cat([tail, codes_chunk_tq], dim=0)
        else:
            codes_in_tq = codes_chunk_tq

        pkv = state.decoder_past_key_values
        if ov > 0 and pkv is not None:
            pkv = self.drop_cache_tail(pkv, ov)

        audio_codes = (
            codes_in_tq.to(self.codec.device).permute(1, 0).unsqueeze(0).contiguous()
        )

        out = self.codec.model.decode(
            audio_codes=audio_codes,
            decoder_past_key_values=pkv,
            return_dict=True,
        )

        wav = out.audio_values
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.reshape(1, -1)

        expected_total = int((ov + n_new) * hop)
        if wav.size(1) >= expected_total:
            wav = wav[:, :expected_total]

        ov_samp = min(int(ov * hop), int(wav.size(1)))
        wav_new = wav[:, ov_samp:]

        state.decoder_past_key_values = getattr(out, "decoder_past_key_values", None)
        state.frames_seen += n_new
        state.samples_emitted += int(wav_new.size(1))

        if overlap_frames > 0:
            keep = min(int(overlap_frames), int(codes_in_tq.size(0)))
            state.tail_codes_tq = codes_in_tq[-keep:].detach()
        else:
            state.tail_codes_tq = None

        return wav_new, state
