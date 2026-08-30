from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROMPT_LEVEL_DB = -19.8
OUTPUT_LEVEL_DB = -23.0
REF_GAIN_LIMIT_DB = 30.0
LIMITER_KNEE = 0.9
MIN_ACTIVE_SECONDS = 0.4

PAUSE_MIN_SECONDS = 0.10
PAUSE_KEEP_SECONDS = 0.15
CROP_FORWARD_SECONDS = 5.0
CROP_BACKWARD_SECONDS = 5.0
MIN_KEEP_FRACTION = 0.75
ROOM_TONE_SECONDS = 0.25
FADE_SECONDS = 0.02

ONSET_THRESHOLD_DB = -45.0
ONSET_OVER_FLOOR_DB = 15.0
ONSET_WINDOW_FRAMES = 6
ONSET_MIN_FRAMES = 5
LEAD_IN_SECONDS = 0.08
SEGMENT_LEAD_SECONDS = 0.30
SEGMENT_SKIP_SECONDS = 0.10
TRAIL_SECONDS = 0.30
GATE_HOLD_SECONDS = 2.0
JOIN_FADE_SECONDS = 0.01


@lru_cache(maxsize=None)
def _resampler(src_sr: int, dst_sr: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(orig_freq=int(src_sr), new_freq=int(dst_sr)).eval()


def resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if int(src_sr) == int(dst_sr):
        return wav
    rs = _resampler(int(src_sr), int(dst_sr))
    if rs.kernel.device != wav.device:
        rs.to(wav.device)
    return rs(wav.float()).to(dtype=wav.dtype)


def load_audio(path: Union[str, Path], sample_rate: int) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return to_mono_resampled(wav, int(sr), int(sample_rate))


def to_mono_resampled(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    wav = wav.detach().float()
    if wav.dim() > 1:
        wav = wav.mean(dim=0 if wav.shape[0] <= wav.shape[1] else 1)
    return resample(wav, int(src_sr), int(dst_sr)).clamp(-1.0, 1.0)


def save_wav(path: Union[str, Path], wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu().float()
    if wav.dim() == 2:
        wav = wav[0] if wav.shape[0] == 1 else wav.mean(dim=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.numpy(), int(sample_rate), subtype="PCM_16")


def _frame_rms(x: torch.Tensor, sample_rate: int, win_s: float, hop_s: float) -> torch.Tensor:
    win, hop = int(sample_rate * win_s), int(sample_rate * hop_s)
    return x.unfold(0, win, hop).pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)


def _pause_runs(quiet: torch.Tensor, min_run: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i, n = 0, int(quiet.numel())
    while i < n:
        if bool(quiet[i]):
            j = i
            while j < n and bool(quiet[j]):
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _finish_with_room_tone(wav: torch.Tensor, sample_rate: int, floor: float) -> torch.Tensor:
    fade = int(FADE_SECONDS * sample_rate)
    out = wav.clone()
    out[..., -fade:] = out[..., -fade:] * torch.linspace(1.0, 0.0, fade, device=wav.device, dtype=wav.dtype)
    tone = torch.randn(int(ROOM_TONE_SECONDS * sample_rate), device=wav.device, dtype=wav.dtype) * float(floor)
    return torch.cat([out, tone], dim=-1)


def crop_on_pause(wav: torch.Tensor, target_seconds: float, sample_rate: int) -> torch.Tensor:
    x = wav.detach().float().reshape(-1)
    win, hop = int(sample_rate * 0.025), int(sample_rate * 0.010)
    total = int(x.numel())
    if total < 4 * win:
        return wav
    rms = _frame_rms(x, sample_rate, 0.025, 0.010)
    floor = float(torch.quantile(rms, 0.1))
    runs = _pause_runs((rms < floor * 4.0).cpu(), max(1, int(round(PAUSE_MIN_SECONDS / 0.010))))
    keep = int(round(PAUSE_KEEP_SECONDS / 0.010))
    target = int(round(float(target_seconds) * sample_rate))

    def cut_at(run: Tuple[int, int]) -> torch.Tensor:
        a, b = run
        return wav[..., : (a + min(b - a, keep)) * hop + win]

    if total <= target:
        if runs and runs[-1][1] * hop + win >= total - hop:
            return wav
        inside = [r for r in runs if r[0] * hop >= int(total * MIN_KEEP_FRACTION)]
        return cut_at(inside[-1]) if inside else _finish_with_room_tone(wav, sample_rate, floor)
    forward = [r for r in runs if target <= r[0] * hop <= target + int(CROP_FORWARD_SECONDS * sample_rate)]
    if forward:
        return cut_at(forward[0])
    backward = [r for r in runs if r[0] * hop < target and r[1] * hop >= target - int(CROP_BACKWARD_SECONDS * sample_rate)]
    if backward:
        return cut_at(backward[-1])
    return _finish_with_room_tone(wav[..., :target], sample_rate, floor)


def speech_level_db(wav: torch.Tensor, sample_rate: int) -> Tuple[float, float]:
    x = wav.detach().float().reshape(-1)
    win, hop = int(sample_rate * 0.025), int(sample_rate * 0.010)
    if int(x.numel()) < win:
        return float(20.0 * math.log10(max(float(x.pow(2).mean().sqrt()), 1e-6))), 0.0
    rms = _frame_rms(x, sample_rate, 0.025, 0.010)
    keep = rms > torch.quantile(rms, 0.2) * 1.5
    active = rms[keep] if int(keep.sum()) > 0 else rms
    return float(20.0 * torch.log10(torch.median(active))), float(active.numel()) * hop / float(sample_rate)


def normalize_reference(wav: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, float]:
    level_db, _ = speech_level_db(wav, sample_rate)
    gain_db = min(max(PROMPT_LEVEL_DB - level_db, 0.0), REF_GAIN_LIMIT_DB)
    peak = float(wav.detach().abs().max())
    if peak > 0.0:
        gain_db = min(gain_db, max(0.0, 20.0 * math.log10(0.95 / peak)))
    return wav * (10.0 ** (gain_db / 20.0)), level_db + gain_db


def output_gain(prompt_level_db: float = PROMPT_LEVEL_DB) -> float:
    return 10.0 ** ((OUTPUT_LEVEL_DB - float(prompt_level_db)) / 20.0)


def match_gain(wav: torch.Tensor, sample_rate: int, target_db: float = OUTPUT_LEVEL_DB) -> float:
    level_db, seconds = speech_level_db(wav, sample_rate)
    if seconds < MIN_ACTIVE_SECONDS:
        return output_gain()
    return 10.0 ** ((target_db - level_db) / 20.0)


def match_level(wav: torch.Tensor, sample_rate: int, target_db: float = OUTPUT_LEVEL_DB) -> torch.Tensor:
    return wav * match_gain(wav, sample_rate, target_db)


def soft_limit(wav: torch.Tensor, knee: float = LIMITER_KNEE) -> torch.Tensor:
    mag = wav.abs()
    over = knee + (1.0 - knee) * torch.tanh((mag - knee) / (1.0 - knee))
    return torch.where(mag > knee, torch.sign(wav) * over, wav)


def _onset_threshold(rms: torch.Tensor) -> float:
    thr = 10.0 ** (ONSET_THRESHOLD_DB / 20.0)
    if int(rms.numel()) >= 30:
        thr = max(thr, float(torch.quantile(rms, 0.1)) * 10.0 ** (ONSET_OVER_FLOOR_DB / 20.0))
    return thr


def speech_onset(wav: torch.Tensor, sample_rate: int) -> Optional[int]:
    x = wav.detach().float().reshape(-1)
    win = int(sample_rate * 0.010)
    if int(x.numel()) < win * ONSET_WINDOW_FRAMES:
        return None
    rms = x[: (x.numel() // win) * win].view(-1, win).pow(2).mean(dim=-1).sqrt()
    above = (rms > _onset_threshold(rms)).float()
    hits = (above.unfold(0, ONSET_WINDOW_FRAMES, 1).sum(dim=-1) >= ONSET_MIN_FRAMES).nonzero()
    if int(hits.numel()) == 0:
        return None
    return int(hits[0]) * win


def energy_onset(wav: torch.Tensor, sample_rate: int, over_floor_db: float = 10.0, min_frames: int = 3) -> Optional[int]:
    x = wav.detach().float().reshape(-1)
    win = int(sample_rate * 0.010)
    if int(x.numel()) < win * min_frames:
        return None
    rms = x[: (x.numel() // win) * win].view(-1, win).pow(2).mean(dim=-1).sqrt()
    if int(rms.numel()) < min_frames:
        return None
    floor = float(torch.quantile(rms, 0.1))
    thr = max(floor * 10.0 ** (over_floor_db / 20.0), 10.0 ** (ONSET_THRESHOLD_DB / 20.0))
    above = (rms > thr).float()
    hits = (above.unfold(0, min_frames, 1).sum(dim=-1) >= min_frames).nonzero()
    if int(hits.numel()) == 0:
        return None
    return int(hits[0]) * win


def energy_fraction(wav: torch.Tensor, sample_rate: int, start: int, end: int, over_floor_db: float = 10.0) -> float:
    x = wav.detach().float().reshape(-1)
    win = int(sample_rate * 0.010)
    if end <= start or int(x.numel()) < win:
        return 0.0
    rms = x[: (x.numel() // win) * win].view(-1, win).pow(2).mean(dim=-1).sqrt()
    floor = float(torch.quantile(rms, 0.1))
    thr = max(floor * 10.0 ** (over_floor_db / 20.0), 10.0 ** (ONSET_THRESHOLD_DB / 20.0))
    seg = rms[start // win : max(start // win + 1, end // win)]
    if int(seg.numel()) == 0:
        return 0.0
    return float((seg > thr).float().mean())


def trim_lead(wav: torch.Tensor, sample_rate: int, lead: float = LEAD_IN_SECONDS, skip: float = 0.0) -> torch.Tensor:
    onset = speech_onset(wav, sample_rate)
    if onset is None:
        return wav
    cut = max(onset - int(lead * sample_rate), int(skip * sample_rate))
    cut = min(cut, max(0, onset - int(0.02 * sample_rate)))
    return wav[..., cut:]


def trim_trail(wav: torch.Tensor, sample_rate: int, trail: float = TRAIL_SECONDS) -> torch.Tensor:
    x = wav.detach().float().reshape(-1)
    win = int(sample_rate * 0.010)
    if int(x.numel()) < win:
        return wav
    rms = x[: (x.numel() // win) * win].view(-1, win).pow(2).mean(dim=-1).sqrt()
    above = (rms > _onset_threshold(rms)).nonzero()
    if int(above.numel()) == 0:
        return wav
    end = min(int(x.numel()), (int(above[-1]) + 1) * win + int(trail * sample_rate))
    return wav[..., :end]


def fade_edges(wav: torch.Tensor, sample_rate: int, fade_in: bool, fade_out: bool, fade_seconds: float = JOIN_FADE_SECONDS) -> torch.Tensor:
    fade = int(fade_seconds * sample_rate)
    if int(wav.shape[-1]) <= 2 * fade:
        return wav
    out = wav.clone()
    ramp = torch.linspace(0.0, 1.0, fade, device=wav.device, dtype=wav.dtype)
    if fade_in:
        out[..., :fade] = out[..., :fade] * ramp
    if fade_out:
        out[..., -fade:] = out[..., -fade:] * ramp.flip(0)
    return out


def join_segments(parts: List[torch.Tensor], sample_rate: int) -> torch.Tensor:
    n = len(parts)
    return torch.cat([fade_edges(p, sample_rate, fade_in=i > 0, fade_out=i + 1 < n) for i, p in enumerate(parts)], dim=-1)
