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
    n_layers_text: int = 2
    dropout: float = 0.05
    pos_emb_max: int = 4096
    max_text_len: int = 2048

    n_layers_ar: int = 6
    ar_kernel: int = 13
    ar_dilation_cycle: Tuple[int, ...] = (1, 2, 4, 1)
    ar_text_attn_freq: int = 2
    min_gen_frames: int = 12

    n_layers_nar: int = 6
    nar_head_dim: int = 256
    nar_kernel_size: int = 11
    nar_dilation_cycle: Tuple[int, ...] = (1, 2, 4, 8)

    stage_B: Tuple[int, int] = (2, 4)
    stage_C: Tuple[int, int] = (5, 8)
    stage_D: Tuple[int, int] = (9, 16)
    stage_E: Tuple[int, int] = (17, 32)

    sv_student_dim: int = 192
    style_strength: float = 1.0

    ref_enc_layers: int = 2
    ref_xattn_heads: int = 2
    ref_xattn_layers: int = 3
    ref_xattn_gmax: float = 0.35
