from __future__ import annotations

import os
import struct
import tempfile
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sopro import SoproTTS
from sopro.audio import device_str
from sopro.constants import TARGET_SR

@dataclass
class ServerConfig:
    repo_id: str = os.getenv("SOPRO_REPO_ID", "samuel-vitorino/sopro")
    revision: Optional[str] = os.getenv("SOPRO_REVISION", None)
    cache_dir: Optional[str] = os.getenv("HF_HOME", None)
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

@app.post("/v1/audio/speech")
async def speech(
    input: str = Form(...),
    stream: bool = Form(False),

    ref_audio: UploadFile = File(...),

    max_frames: int = Form(400),
    top_p: float = Form(0.9),
    temperature: float = Form(1.05),

    anti_loop: bool = Form(True),
    use_prefix: bool = Form(True),
    prefix_sec: Optional[float] = Form(None),
    style_strength: float = Form(1.0),
    ref_seconds: Optional[float] = Form(None),

    use_stop_head: bool = Form(True),
    stop_threshold: Optional[float] = Form(None),
    stop_patience: Optional[int] = Form(None),
):
    if not input.strip():
        raise HTTPException(status_code=400, detail="`input` must be non-empty.")

    suffix = os.path.splitext(ref_audio.filename or "")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tmp_path = tf.name
        data = await ref_audio.read()
        await ref_audio.close()
        if not data:
            raise HTTPException(status_code=400, detail="Empty `ref_audio` upload.")
        tf.write(data)

    tts = get_tts()

    max_frames = int(max(1, min(int(max_frames), 2000)))
    top_p = float(max(0.01, min(float(top_p), 1.0)))
    temperature = float(max(0.05, min(float(temperature), 3.0)))
    style_strength = float(max(0.0, min(float(style_strength), 3.0)))

    if not stream:
        try:
            with _gen_lock:
                wav = tts.synthesize(
                    input,
                    ref_audio_path=tmp_path,
                    max_frames=max_frames,
                    top_p=top_p,
                    temperature=temperature,
                    anti_loop=anti_loop,
                    use_prefix=use_prefix,
                    prefix_sec_fixed=prefix_sec,
                    style_strength=style_strength,
                    ref_seconds=ref_seconds,
                    use_stop_head=use_stop_head,
                    stop_threshold=stop_threshold,
                    stop_patience=stop_patience,
                )

            wav_bytes = wav_bytes_from_float(wav, sr=TARGET_SR)
            return Response(content=wav_bytes, media_type="audio/wav")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield stream_header(TARGET_SR, 1)

            with _gen_lock:
                for chunk in tts.stream(
                    input,
                    ref_audio_path=tmp_path,
                    max_frames=max_frames,
                    top_p=top_p,
                    temperature=temperature,
                    anti_loop=anti_loop,
                    use_prefix=use_prefix,
                    prefix_sec_fixed=prefix_sec,
                    style_strength=style_strength,
                    ref_seconds=ref_seconds,
                    chunk_frames=CFG.chunk_size,
                    use_stop_head=use_stop_head,
                    stop_threshold=stop_threshold,
                    stop_patience=stop_patience,
                ):
                    payload = float_to_pcm16le(chunk)
                    if payload:
                        yield frame(payload)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="application/octet-stream")
