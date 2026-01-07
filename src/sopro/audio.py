from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import TARGET_SR

try:
    import torchaudio
    import torchaudio.functional as AF
except Exception:
    torchaudio = None
    AF = None

try:
    import soundfile as sf
except Exception:
    sf = None


def device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def trim_silence_energy(
    wav: torch.Tensor,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    thresh_db_floor: float = -40.0,
    prepad_ms: float = 30.0,
    postpad_ms: float = 30.0,
    min_keep_sec: float = 0.5,
) -> torch.Tensor:
    orig_1d = wav.ndim == 1
    if orig_1d:
        wav = wav.unsqueeze(0)

    C, T = wav.shape
    if T == 0:
        return wav.squeeze(0) if orig_1d else wav
    if T < int(sr * 0.1):
        return wav.squeeze(0) if orig_1d else wav

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if T < frame_len:
        return wav.squeeze(0) if orig_1d else wav

    mono = wav.mean(dim=0, keepdim=True)
    frames = mono.unfold(-1, frame_len, hop)
    energy = frames.pow(2).mean(dim=-1).squeeze(0)

    eps = 1e-10
    energy_db = 10.0 * torch.log10(energy + eps)
    max_db = float(energy_db.max().item())

    rel_thresh = max_db + thresh_db_floor
    thresh_db = max(rel_thresh, thresh_db_floor)

    voiced = energy_db > thresh_db
    idx = torch.nonzero(voiced, as_tuple=False)
    if idx.numel() == 0:
        return wav.squeeze(0) if orig_1d else wav

    first_frame = int(idx[0, 0].item())
    last_frame = int(idx[-1, 0].item())

    prepad_samples = int(sr * prepad_ms / 1000.0)
    postpad_samples = int(sr * postpad_ms / 1000.0)

    start = max(0, first_frame * hop - prepad_samples)

    end = min(T, last_frame * hop + frame_len + postpad_samples)

    min_keep = int(min_keep_sec * sr)
    if end <= start or (end - start) < min_keep:
        return wav.squeeze(0) if orig_1d else wav

    out = wav[:, start:end]
    return out.squeeze(0) if orig_1d else out


def load_audio_file(path: str) -> Tuple[torch.Tensor, int]:
    if sf is not None:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1)
    elif torchaudio is not None:
        wav, sr = torchaudio.load(path)
        if wav.dtype != torch.float32:
            if wav.dtype == torch.int16:
                wav = wav.float() / (2**15)
            else:
                wav = wav.float()
    else:
        raise RuntimeError("Install 'soundfile' or 'torchaudio' to read audio.")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample(
    wav: torch.Tensor, sr_in: int, sr_out: int, device: Optional[str] = None
) -> torch.Tensor:
    device = device or device_str()
    wav = wav.to(device)
    if sr_in == sr_out:
        return wav
    if sr_in != sr_out and AF is None:
        raise RuntimeError("Resampling requires torchaudio. pip install torchaudio")
    return AF.resample(wav, sr_in, sr_out)


def save_audio(path: str, wav: torch.Tensor, sr: int = TARGET_SR) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    wav = wav.detach().cpu()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        pass
    elif wav.ndim == 3:
        wav = wav[0]
    else:
        raise ValueError(f"Expected wav with 1-3 dims, got shape {tuple(wav.shape)}")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sf is not None:
        sf.write(path, wav[0].numpy(), sr)
        return

    if torchaudio is not None:
        torchaudio.save(path, wav, sample_rate=sr)
        return

    raise RuntimeError("Install 'soundfile' or 'torchaudio' to write audio.")


def center_crop_audio(wav: torch.Tensor, win_samples: int) -> torch.Tensor:
    if win_samples <= 0:
        return wav
    T = int(wav.shape[-1])
    if T <= win_samples:
        return wav
    s = (T - win_samples) // 2
    return wav[..., s : s + win_samples]
