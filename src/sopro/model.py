from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torchaudio

from sopro import audio as audio_ops
from sopro.config import GenerationConfig, SoproConfig
from sopro.encoders.semantic import SemanticEncoder
from sopro.encoders.speaker import SpeakerEncoder
from sopro.hub import MODEL_FILE, SEMANTIC_ENCODER_FILE, SPEAKER_ENCODER_FILE, TOKENIZER_FILE, VOCODER_FILE, load_config, load_weights, resolve_artifacts
from sopro.nn.model import SoproModel
from sopro.nn.quant import quantize_ar_int8
from sopro.streaming import PromptState, StreamSession, build_prompt_state
from sopro.text import TextTokenizer, split_text
from sopro.vocoder import Vocoder

DECODE_CONTEXT_FRAMES = 32
FINAL_FADE_SECONDS = 0.08
PROGRESS_CHUNK_TOKENS = 24


@dataclass
class Reference:
    cond_vec: torch.Tensor
    semantic_tokens: torch.Tensor
    mel: torch.Tensor
    level_db: float = -19.8
    prompt_states: Dict[Tuple[int, int], PromptState] = field(default_factory=dict)

    def to(self, device: torch.device) -> "Reference":
        if self.cond_vec.device == device:
            return self
        return Reference(self.cond_vec.to(device), self.semantic_tokens.to(device), self.mel.to(device), self.level_db, {k: v.to(device) for k, v in self.prompt_states.items()})


class SoproTTS:
    def __init__(
        self,
        config: SoproConfig,
        model: SoproModel,
        semantic_encoder: SemanticEncoder,
        speaker_encoder: SpeakerEncoder,
        vocoder: Vocoder,
        tokenizer: TextTokenizer,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.semantic_encoder = semantic_encoder
        self.speaker_encoder = speaker_encoder
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.device = device
        self.sample_rate = int(config.sample_rate)
        mcfg = config.model
        self.hop_ratio = int(semantic_encoder.token_samples) // int(mcfg.acoustic_mel_hop_length)
        self.mel_mean = torch.tensor(mcfg.acoustic_mel_mean, device=device, dtype=torch.float32).view(1, -1, 1)
        self.mel_std = torch.tensor(mcfg.acoustic_mel_std, device=device, dtype=torch.float32).view(1, -1, 1)

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: Union[str, Path] = "samuel-vitorino/sopro-v2-turbo",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> "SoproTTS":
        device = torch.device(device)
        if quantization not in (None, "int8"):
            raise ValueError(f"unsupported quantization {quantization!r}; expected None or 'int8'")
        if quantization == "int8" and device.type != "cpu":
            raise ValueError("int8 quantization is supported on cpu only")
        root = resolve_artifacts(repo_or_path, revision=revision, cache_dir=cache_dir)
        config = load_config(root)
        model = SoproModel(config.model)
        model.load_state_dict(load_weights(root, MODEL_FILE, torch.device("cpu")))
        model = model.to(device=device, dtype=dtype).eval()
        if quantization == "int8":
            quantize_ar_int8(model)
        semantic_encoder = SemanticEncoder(config.semantic_encoder)
        semantic_encoder.load_state_dict(load_weights(root, SEMANTIC_ENCODER_FILE, torch.device("cpu")))
        speaker_encoder = SpeakerEncoder(config.speaker_encoder)
        speaker_encoder.load_state_dict(load_weights(root, SPEAKER_ENCODER_FILE, torch.device("cpu")))
        vocoder = Vocoder(config.vocoder)
        vocoder.load_state_dict(load_weights(root, VOCODER_FILE, torch.device("cpu")))
        for module in (semantic_encoder, speaker_encoder, vocoder):
            module.to(device).eval()
            for p in module.parameters():
                p.requires_grad_(False)
        for p in model.parameters():
            p.requires_grad_(False)
        tokenizer = TextTokenizer(str(root / TOKENIZER_FILE))
        return cls(config, model, semantic_encoder, speaker_encoder, vocoder, tokenizer, device)

    @property
    def generation(self) -> GenerationConfig:
        return self.config.generation

    @torch.no_grad()
    def prepare_reference(
        self,
        ref_audio_path: Optional[Union[str, Path]] = None,
        ref_audio: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        seconds: Optional[float] = None,
        stream: bool = False,
    ) -> Reference:
        if ref_audio_path is not None:
            wav = audio_ops.load_audio(ref_audio_path, self.sample_rate)
        elif ref_audio is not None:
            wav = audio_ops.to_mono_resampled(ref_audio, int(sample_rate or self.sample_rate), self.sample_rate)
        else:
            raise ValueError("provide ref_audio_path or ref_audio")
        wav = audio_ops.crop_on_pause(wav.to(self.device), float(self.generation.ref_seconds if seconds is None else seconds), self.sample_rate).unsqueeze(0)
        wav, level_db = audio_ops.normalize_reference(wav, self.sample_rate)
        wav16 = torchaudio.functional.resample(wav, self.sample_rate, int(self.config.speaker_encoder.sample_rate))
        spk = self.speaker_encoder(wav16)
        cond_vec = self.model.build_condition(*(spk[k].to(dtype=self.model.dtype) for k in ("id_emb", "style_emb", "style_ctrl")))
        tokens = self.semantic_encoder.encode(wav)
        mel = (self.vocoder.mel(wav) - self.mel_mean) / self.mel_std
        ref = Reference(cond_vec=cond_vec, semantic_tokens=tokens, mel=mel, level_db=float(level_db))
        if stream:
            self._prompt_state(ref, int(self.generation.steps), int(self.generation.stream_chunk_frames))
        return ref

    def _prompt_state(self, ref: Reference, steps: int, chunk_frames: int) -> PromptState:
        key = (int(steps), int(chunk_frames))
        state = ref.prompt_states.get(key)
        if state is None:
            state = build_prompt_state(self.model, ref.semantic_tokens, ref.mel, ref.cond_vec, int(steps), int(chunk_frames), self.hop_ratio)
            ref.prompt_states[key] = state
        return state

    def _resolve_reference(self, ref: Optional[Reference], ref_audio_path: Optional[Union[str, Path]]) -> Reference:
        if ref is not None:
            return ref.to(self.device)
        if ref_audio_path is None:
            raise ValueError("provide ref or ref_audio_path")
        return self.prepare_reference(ref_audio_path=ref_audio_path)

    def _steps(self, seconds: float) -> int:
        return max(1, int(math.ceil(float(seconds) * float(self.sample_rate) / float(self.semantic_encoder.token_samples))))

    def _semantic_stream(self, text: str, lang: Optional[str], ref: Reference, chunk_size: int, max_seconds: float, min_seconds: float, temperature: float, top_p: float, top_k: int, prompt_tokens: Optional[torch.Tensor] = None) -> Iterator[torch.Tensor]:
        gen = self.generation
        text_ids = self.tokenizer.encode_tensor(text, lang, self.device)
        style_tokens = ref.semantic_tokens[:, : int(gen.style_tokens)]
        if prompt_tokens is None:
            prompt_tokens = ref.semantic_tokens[:, : int(gen.prompt_tokens)] if int(gen.prompt_tokens) > 0 else None
        return self.model.stream_semantic_tokens(
            text_ids,
            style_tokens,
            prompt_tokens,
            chunk_size=int(chunk_size),
            max_steps=self._steps(max_seconds),
            min_steps=self._steps(min_seconds),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        ref: Optional[Reference] = None,
        ref_audio_path: Optional[Union[str, Path]] = None,
        lang: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        steps: Optional[int] = None,
        max_seconds: Optional[float] = None,
        min_seconds: Optional[float] = None,
        on_tokens: Optional[Callable[[int], None]] = None,
    ) -> torch.Tensor:
        gen = self.generation
        ref = self._resolve_reference(ref, ref_audio_path)
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        top_k = gen.top_k if top_k is None else top_k
        steps = gen.steps if steps is None else steps
        max_seconds = gen.max_seconds if max_seconds is None else max_seconds
        min_seconds = gen.min_seconds if min_seconds is None else min_seconds
        parts = []
        carry: Optional[torch.Tensor] = None
        done = 0
        for seg in split_text(text, int(gen.max_segment_chars)):
            seg_cb = None
            if on_tokens is not None:
                base = done
                seg_cb = lambda n, base=base: on_tokens(base + n)
            wav, tokens = self._synthesize_segment(seg, ref, lang, temperature, top_p, top_k, steps, max_seconds, min_seconds, prompt_tokens=carry, on_tokens=seg_cb)
            done += int(tokens.shape[-1])
            if int(tokens.shape[-1]) > 0 and int(gen.prompt_tokens) > 0:
                carry = tokens[:, -int(gen.prompt_tokens) :]
            parts.append(wav)
        parts = [p for p in parts if int(p.shape[-1]) > 0]
        if not parts:
            return torch.zeros(int(self.semantic_encoder.token_samples), device=self.device)
        gain = audio_ops.output_gain(ref.level_db)
        n = len(parts)
        trimmed = []
        for i, p in enumerate(parts):
            p = p * gain
            p = audio_ops.trim_lead(p, self.sample_rate) if i == 0 else audio_ops.trim_lead(p, self.sample_rate, audio_ops.SEGMENT_LEAD_SECONDS, audio_ops.SEGMENT_SKIP_SECONDS)
            trimmed.append(audio_ops.trim_trail(p, self.sample_rate))
        out = audio_ops.soft_limit(audio_ops.join_segments(trimmed, self.sample_rate))
        return audio_ops.fade_edges(out, self.sample_rate, fade_in=False, fade_out=True, fade_seconds=FINAL_FADE_SECONDS)

    def _synthesize_segment(self, text: str, ref: Reference, lang: Optional[str], temperature: float, top_p: float, top_k: int, steps: int, max_seconds: float, min_seconds: float, prompt_tokens: Optional[torch.Tensor] = None, on_tokens: Optional[Callable[[int], None]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = []
        count = 0
        chunk_size = PROGRESS_CHUNK_TOKENS if on_tokens is not None else self._steps(max_seconds)
        for chunk in self._semantic_stream(text, lang, ref, chunk_size, max_seconds, min_seconds, temperature, top_p, top_k, prompt_tokens=prompt_tokens):
            chunks.append(chunk)
            if on_tokens is not None:
                count += int(chunk.shape[-1])
                on_tokens(count)
        empty = ref.semantic_tokens.new_zeros((1, 0))
        if not chunks:
            return torch.zeros(0, device=self.device), empty
        tokens = torch.cat(chunks, dim=1)
        n_tok = int(tokens.shape[-1])
        target_length = n_tok * int(self.semantic_encoder.token_samples)
        p_mel = int(ref.mel.shape[-1])
        n_mels = int(ref.mel.shape[1])
        t_max = p_mel + n_tok * self.hop_ratio
        cond_mel = ref.mel.new_zeros((1, n_mels, t_max))
        cond_mask = ref.mel.new_zeros((1, 1, t_max))
        cond_mel[:, :, :p_mel] = ref.mel
        cond_mask[:, :, :p_mel] = 1.0
        x0 = torch.randn((1, n_mels, t_max), device=self.device, dtype=ref.cond_vec.dtype)
        mel = self.model.acoustic_head.solve(x0, torch.cat([ref.semantic_tokens, tokens], dim=1), ref.cond_vec, cond_mel, cond_mask, int(steps))
        ctx = min(DECODE_CONTEXT_FRAMES, p_mel)
        hop = int(self.vocoder.hop_length)
        wav = self.vocoder.decode(mel[:, :, p_mel - ctx :].float() * self.mel_std + self.mel_mean)[0]
        return wav[ctx * hop : ctx * hop + target_length], tokens

    @torch.inference_mode()
    def stream(
        self,
        text: str,
        ref: Optional[Reference] = None,
        ref_audio_path: Optional[Union[str, Path]] = None,
        lang: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        steps: Optional[int] = None,
        max_seconds: Optional[float] = None,
        min_seconds: Optional[float] = None,
        chunk_frames: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        gen = self.generation
        ref = self._resolve_reference(ref, ref_audio_path)
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        top_k = gen.top_k if top_k is None else top_k
        steps = gen.steps if steps is None else steps
        max_seconds = gen.max_seconds if max_seconds is None else max_seconds
        min_seconds = gen.min_seconds if min_seconds is None else min_seconds
        chunk_frames = int(gen.stream_chunk_frames if chunk_frames is None else chunk_frames)
        if chunk_frames < 64:
            raise ValueError("chunk_frames must be at least 64")
        if chunk_frames % self.hop_ratio != 0:
            raise ValueError(f"chunk_frames must be a multiple of {self.hop_ratio}")
        segments = split_text(text, int(gen.max_segment_chars))
        carry: Optional[torch.Tensor] = None
        for i, segment in enumerate(segments):
            tokens = yield from self._stream_segment(segment, ref, lang, temperature, top_p, top_k, steps, max_seconds, min_seconds, chunk_frames, first=i == 0, last=i + 1 == len(segments), prompt_tokens=carry)
            if tokens is not None and int(tokens.shape[-1]) > 0 and int(gen.prompt_tokens) > 0:
                carry = tokens[:, -int(gen.prompt_tokens) :]

    def _stream_segment(self, text: str, ref: Reference, lang: Optional[str], temperature: float, top_p: float, top_k: int, steps: int, max_seconds: float, min_seconds: float, chunk_frames: int, first: bool = True, last: bool = True, prompt_tokens: Optional[torch.Tensor] = None) -> Iterator[torch.Tensor]:
        session = StreamSession(
            self.model,
            self.vocoder,
            ref.semantic_tokens,
            ref.mel,
            ref.cond_vec,
            self.mel_mean,
            self.mel_std,
            chunk_frames=chunk_frames,
            hop_ratio=self.hop_ratio,
            steps=int(steps),
            warmup_frames=min(DECODE_CONTEXT_FRAMES, int(ref.mel.shape[-1])),
            max_frames=int(ref.mel.shape[-1]) + self._steps(max_seconds) * self.hop_ratio,
            prompt_state=self._prompt_state(ref, int(steps), chunk_frames),
        )
        emitted = 0
        gain = audio_ops.output_gain(ref.level_db)
        lead = audio_ops.LEAD_IN_SECONDS if first else audio_ops.SEGMENT_LEAD_SECONDS
        skip = 0.0 if first else audio_ops.SEGMENT_SKIP_SECONDS
        pending: Optional[torch.Tensor] = torch.zeros(0, device=self.device)
        pending_offset = 0

        def gate(audio: torch.Tensor) -> Optional[torch.Tensor]:
            nonlocal pending, pending_offset
            if pending is None:
                return audio
            pending = torch.cat([pending, audio], dim=-1)
            onset = audio_ops.speech_onset(pending, self.sample_rate)
            if onset is None:
                keep = int(max(float(lead), audio_ops.GATE_HOLD_SECONDS) * self.sample_rate)
                if int(pending.shape[-1]) > keep:
                    pending_offset += int(pending.shape[-1]) - keep
                    pending = pending[-keep:]
                return None
            abs_onset = pending_offset + onset
            cut = max(abs_onset - int(lead * self.sample_rate), int(skip * self.sample_rate))
            cut = min(cut, max(0, abs_onset - int(0.02 * self.sample_rate)))
            guard = audio_ops.energy_onset(pending, self.sample_rate)
            if guard is not None and guard < onset:
                near = onset - guard <= int(0.05 * self.sample_rate)
                if near or audio_ops.energy_fraction(pending, self.sample_rate, guard, onset) >= 0.35:
                    cut = min(cut, max(int(skip * self.sample_rate), pending_offset + guard - int(0.05 * self.sample_rate)))
            out = pending[max(0, cut - pending_offset) :]
            pending = None
            return audio_ops.fade_edges(out, self.sample_rate, fade_in=not first, fade_out=False)

        for sem_chunk in self._semantic_stream(text, lang, ref, chunk_frames // self.hop_ratio, max_seconds, min_seconds, temperature, top_p, top_k, prompt_tokens=prompt_tokens):
            audio = session.push(sem_chunk)
            if audio is not None and int(audio.shape[-1]) > 0:
                emitted += int(audio.shape[-1])
                out = gate(audio * gain)
                if out is not None and int(out.shape[-1]) > 0:
                    yield audio_ops.soft_limit(out)
        n_tok = int(session.tokens.shape[-1])
        if n_tok == 0:
            return None
        tail = session.finish()[: max(0, n_tok * int(self.semantic_encoder.token_samples) - emitted)]
        if int(tail.shape[-1]) > 0:
            out = gate(tail * gain)
            if out is not None and int(out.shape[-1]) > 0:
                out = audio_ops.fade_edges(audio_ops.trim_trail(out, self.sample_rate), self.sample_rate, fade_in=False, fade_out=True,
                                           fade_seconds=FINAL_FADE_SECONDS if last else audio_ops.JOIN_FADE_SECONDS)
                yield audio_ops.soft_limit(out)
        elif pending is not None and int(pending.shape[-1]) > 0:
            yield audio_ops.soft_limit(audio_ops.fade_edges(pending, self.sample_rate, fade_in=False, fade_out=last, fade_seconds=FINAL_FADE_SECONDS))
        return session.tokens


    def save_wav(self, path: Union[str, Path], wav: torch.Tensor) -> None:
        audio_ops.save_wav(path, wav, self.sample_rate)
