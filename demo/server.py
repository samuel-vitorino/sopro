from __future__ import annotations

import os
import struct
import tempfile
import threading
import hashlib
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sopro import SoproTTS
from sopro.model import PreparedReference
from sopro.audio import device_str
from sopro.constants import TARGET_SR

try:
    torch.serialization.add_safe_globals([PreparedReference])
except Exception:
    pass

@dataclass
class ServerConfig:
    repo_id: str = os.getenv("SOPRO_REPO_ID", "samuel-vitorino/sopro")
    revision: Optional[str] = os.getenv("SOPRO_REVISION", None)
    cache_dir: Optional[str] = os.getenv("HF_HOME", None)
    ref_cache_dir: str = os.getenv("SOPRO_REF_CACHE_DIR", "./ref_sv_cache")
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)
    device: str = os.getenv("SOPRO_DEVICE", device_str())
    chunk_size: int = int(os.getenv("SOPRO_CHUNK_SIZE", 16))

CFG = ServerConfig()

_app_lock = threading.Lock()
_tts: Optional[SoproTTS] = None

def get_tts() -> SoproTTS:
    global _tts
    if _tts is not None:
        return _tts
    with _app_lock:
        if _tts is None:
            _tts = SoproTTS.from_pretrained(
                CFG.repo_id,
                revision=CFG.revision,
                cache_dir=CFG.cache_dir,
                token=CFG.hf_token,
                device=CFG.device,
            )
    return _tts

_gen_lock = threading.Lock()
_ref_lock = threading.Lock()

def _effective_ref_seconds(ref_seconds: Optional[float]) -> float:
    return float(ref_seconds) if (ref_seconds is not None and ref_seconds > 0) else 12.0

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sv_cache_path(ref_hash: str, ref_seconds: float) -> str:
    os.makedirs(CFG.ref_cache_dir, exist_ok=True)
    return os.path.join(CFG.ref_cache_dir, f"{ref_hash}_rs{ref_seconds:.3f}.pt")

def _ref_to_cpu(ref: PreparedReference) -> PreparedReference:
    ref.ref_tokens_btq = ref.ref_tokens_btq.detach().cpu()
    ref.sv_ref = ref.sv_ref.detach().cpu()
    ref.ref_seq = ref.ref_seq.detach().cpu()
    ref.ref_kv_caches = [
        {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        for d in ref.ref_kv_caches
    ]
    return ref

def _ref_to_device(ref: PreparedReference, device: torch.device) -> PreparedReference:
    ref.ref_tokens_btq = ref.ref_tokens_btq.to(device)
    ref.sv_ref = ref.sv_ref.to(device)
    ref.ref_seq = ref.ref_seq.to(device)
    ref.ref_kv_caches = [
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}
        for d in ref.ref_kv_caches
    ]
    return ref

@torch.inference_mode()
def get_or_compute_ref(tts: SoproTTS, ref_bytes: bytes, *, suffix: str, ref_seconds: Optional[float]) -> PreparedReference:
    rs = _effective_ref_seconds(ref_seconds)
    h = sha256_bytes(ref_bytes)
    p = sv_cache_path(h, rs)

    if os.path.exists(p):
        ref = torch.load(p, map_location="cpu", weights_only=True)
        return _ref_to_device(ref, tts.device)

    with _ref_lock:
        if os.path.exists(p):
            ref = torch.load(p, map_location="cpu", weights_only=True)
            return _ref_to_device(ref, tts.device)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as tf:
            tmp_path = tf.name
            tf.write(ref_bytes)

        try:
            with _gen_lock:
                ref = tts.prepare_reference(ref_audio_path=tmp_path, ref_seconds=rs)
            torch.save(_ref_to_cpu(ref), p)
            return _ref_to_device(ref, tts.device)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def float_to_pcm16le(wav_1xt: torch.Tensor) -> bytes:
    if wav_1xt.ndim == 1:
        wav_1xt = wav_1xt.unsqueeze(0)
    wav = wav_1xt.detach().cpu().clamp(-1.0, 1.0)
    pcm = (wav * 32767.0).to(torch.int16).numpy()
    return pcm.tobytes(order="C")

def wav_bytes_from_float(wav_1xt: torch.Tensor, sr: int) -> bytes:
    import io
    import wave
    pcm_bytes = float_to_pcm16le(wav_1xt)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm_bytes)
    return bio.getvalue()

MAGIC = b"SPRO"
def stream_header(sr: int, channels: int) -> bytes:
    return MAGIC + struct.pack("<II", int(sr), int(channels))

def frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload

app = FastAPI(title="SoproTTS Demo API", version="0.1")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
def _startup_load_model() -> None:
    get_tts()

@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/v1/reference/cache")
async def cache_reference(
    ref_audio: UploadFile = File(...),
    ref_seconds: Optional[float] = Form(None),
):
    suffix = os.path.splitext(ref_audio.filename or "")[-1] or ".wav"
    data = await ref_audio.read()
    await ref_audio.close()
    if not data:
        raise HTTPException(status_code=400, detail="Empty `ref_audio` upload.")

    tts = get_tts()

    rs = _effective_ref_seconds(ref_seconds)
    rid = sha256_bytes(data)

    _ = get_or_compute_ref(tts, data, suffix=suffix, ref_seconds=rs)

    return {"ref_id": rid, "ref_seconds": rs}

@app.post("/v1/audio/speech")
async def speech(
    input: str = Form(...),
    stream: bool = Form(False),

    ref_id: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),

    max_frames: int = Form(400),
    top_p: float = Form(0.9),
    temperature: float = Form(1.05),

    anti_loop: bool = Form(True),
    style_strength: float = Form(1.2),
    ref_seconds: Optional[float] = Form(None),
):
    if not input.strip():
        raise HTTPException(status_code=400, detail="`input` must be non-empty.")
    
    tts = get_tts()
    rs = _effective_ref_seconds(ref_seconds)

    if (ref_id is None) == (ref_audio is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of `ref_id` or `ref_audio`.")

    if ref_id is not None:
        p = sv_cache_path(ref_id, rs)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail="Cached reference not found. Cache it first.")
        ref = torch.load(p, map_location="cpu", weights_only=True)
        ref = _ref_to_device(ref, tts.device)
    else:
        suffix = os.path.splitext(ref_audio.filename or "")[-1] or ".wav"
        data = await ref_audio.read()
        await ref_audio.close()
        if not data:
            raise HTTPException(status_code=400, detail="Empty `ref_audio` upload.")
        ref = get_or_compute_ref(tts, data, suffix=suffix, ref_seconds=rs)

    max_frames = int(max(1, min(int(max_frames), 2000)))
    top_p = float(max(0.01, min(float(top_p), 1.0)))
    temperature = float(max(0.05, min(float(temperature), 3.0)))
    style_strength = float(max(0.0, min(float(style_strength), 3.0)))

    if not stream:
        with _gen_lock:
            wav = tts.synthesize(
                input,
                ref=ref,
                max_frames=max_frames,
                top_p=top_p,
                temperature=temperature,
                anti_loop=anti_loop,
                style_strength=style_strength,
            )

        wav_bytes = wav_bytes_from_float(wav, sr=TARGET_SR)
        return Response(content=wav_bytes, media_type="audio/wav")

    async def gen() -> AsyncIterator[bytes]:
        yield stream_header(TARGET_SR, 1)

        with _gen_lock:
            for chunk in tts.stream(
                input,
                ref=ref,
                max_frames=max_frames,
                top_p=top_p,
                temperature=temperature,
                anti_loop=anti_loop,
                style_strength=style_strength,
                chunk_frames=CFG.chunk_size,
            ):
                payload = float_to_pcm16le(chunk)
                if payload:
                    yield frame(payload)

    return StreamingResponse(gen(), media_type="application/octet-stream")
