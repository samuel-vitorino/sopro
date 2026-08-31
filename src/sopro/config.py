from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


def _from_dict(cls, data: Optional[Dict[str, Any]]):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in allowed})


@dataclass
class ModelConfig:
    latent_dim: int = 1280
    semantic_vocab_size: int = 4375
    text_vocab_size: int = 8192
    max_text_len: int = 2048
    cond_in_dim: int = 328
    cond_hidden_dim: int = 512
    ar_model_dim: int = 512
    ar_blocks: int = 12
    ar_heads: int = 8
    ar_kv_heads: Optional[int] = None
    ar_ffn_mult: float = 4.0
    ar_qk_rms_norm: bool = True
    style_prefix_tokens: int = 8
    acoustic_time_embed_dim: int = 256
    acoustic_sway_sampling_coef: float = -1.0
    acoustic_upsampler_kernel_size: int = 3
    acoustic_dit_dim: int = 512
    acoustic_dit_depth: int = 8
    acoustic_dit_heads: int = 8
    acoustic_dit_dim_head: int = 64
    acoustic_dit_ff_mult: float = 2.0
    acoustic_mu_dim: Optional[int] = None
    acoustic_spk_dim: int = 80
    acoustic_pre_lookahead_frames: int = 3
    acoustic_pos_kernel_size: int = 31
    acoustic_sigma_min: float = 1.0e-6
    acoustic_num_left_chunks: int = -1
    acoustic_mel_n_mels: int = 100
    acoustic_mel_hop_length: int = 256
    acoustic_mel_mean: Optional[List[float]] = None
    acoustic_mel_std: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.ar_kv_heads is None:
            self.ar_kv_heads = self.ar_heads
        if self.acoustic_mu_dim is None:
            self.acoustic_mu_dim = self.acoustic_mel_n_mels

    @property
    def semantic_bos_id(self) -> int:
        return self.semantic_vocab_size

    @property
    def semantic_eos_id(self) -> int:
        return self.semantic_vocab_size + 1


@dataclass
class SemanticEncoderConfig:
    n_mels: int = 80
    d_model: int = 512
    layers: int = 6
    heads: int = 8
    ffn_dim: int = 2048
    max_positions: int = 1500
    fsq_levels: List[int] = field(default_factory=lambda: [7, 5, 5, 5, 5])
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    token_samples_24k: int = 1024


@dataclass
class SpeakerEncoderConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 20.0
    f_max: float = 7600.0
    mel_log_floor: float = 1e-5
    stem_channels: int = 128
    stage_channels: List[int] = field(default_factory=lambda: [160, 192, 224])
    blocks_per_stage: List[int] = field(default_factory=lambda: [4, 4, 4])
    dilation_cycle: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    depthwise_kernel_size: int = 5
    se_reduction: int = 8
    id_emb_dim: int = 192
    style_emb_dim: int = 128
    style_ctrl_dim: int = 8
    id_head_hidden: int = 256
    style_head_hidden: int = 256
    attn_hidden: int = 128


@dataclass
class VocoderConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 14
    max_magnitude: float = 100.0
    band_limit_hz: float = 10900.0
    causal: bool = False
    lookahead_frames: int = 0
    block_lookaheads: Optional[List[int]] = None


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 25
    steps: int = 2
    max_seconds: float = 30.0
    min_seconds: float = 0.4
    max_segment_chars: int = 300
    ref_seconds: float = 15.0
    style_tokens: int = 160
    prompt_tokens: int = 120
    stream_chunk_frames: int = 64


@dataclass
class SoproConfig:
    sample_rate: int = 24000
    model: ModelConfig = field(default_factory=ModelConfig)
    semantic_encoder: SemanticEncoderConfig = field(default_factory=SemanticEncoderConfig)
    speaker_encoder: SpeakerEncoderConfig = field(default_factory=SpeakerEncoderConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    vocoder_streaming: VocoderConfig = field(default_factory=lambda: VocoderConfig(causal=True))
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoproConfig":
        return cls(
            sample_rate=int(data.get("sample_rate", 24000)),
            model=_from_dict(ModelConfig, data.get("model")),
            semantic_encoder=_from_dict(SemanticEncoderConfig, data.get("semantic_encoder")),
            speaker_encoder=_from_dict(SpeakerEncoderConfig, data.get("speaker_encoder")),
            vocoder=_from_dict(VocoderConfig, data.get("vocoder")),
            vocoder_streaming=_from_dict(VocoderConfig, data.get("vocoder_streaming")),
            generation=_from_dict(GenerationConfig, data.get("generation")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
