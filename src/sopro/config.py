from dataclasses import dataclass
from typing import Tuple

from sopro.constants import TARGET_SR


@dataclass
class SoproTTSConfig:
    num_codebooks: int = 32
    codebook_size: int = 2048
    mimi_fps: float = 12.5
    max_frames: int = 400
    audio_sr: int = TARGET_SR

    d_model: int = 384
    n_layers_text: int = 4
    n_layers_ar: int = 6
    n_layers_nar: int = 6
    dropout: float = 0.05

    pos_emb_max: int = 4096
    max_text_len: int = 2048

    nar_head_dim: int = 256

    use_stop_head: bool = True
    stop_threshold: float = 0.8
    stop_patience: int = 5
    min_gen_frames: int = 12

    stage_B: Tuple[int, int] = (2, 4)
    stage_C: Tuple[int, int] = (5, 8)
    stage_D: Tuple[int, int] = (9, 16)
    stage_E: Tuple[int, int] = (17, 32)

    ar_lookback: int = 4
    ar_kernel: int = 13
    ar_dilation_cycle: Tuple[int, ...] = (1, 2, 4, 1)

    ar_text_attn_freq: int = 2

    ref_attn_heads: int = 2
    ref_seconds_max: float = 12.0

    preprompt_sec_max: float = 4.0

    sv_student_dim: int = 192
    style_strength: float = 1.0
