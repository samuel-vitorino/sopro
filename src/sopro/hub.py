from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from safetensors.torch import load_file

from sopro.config import SoproConfig

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
SEMANTIC_ENCODER_FILE = "semantic_encoder.safetensors"
SPEAKER_ENCODER_FILE = "speaker_encoder.safetensors"
VOCODER_FILE = "vocoder.safetensors"
TOKENIZER_FILE = "tokenizer.model"
ARTIFACT_FILES = (CONFIG_FILE, MODEL_FILE, SEMANTIC_ENCODER_FILE, SPEAKER_ENCODER_FILE, VOCODER_FILE, TOKENIZER_FILE)


def resolve_artifacts(repo_or_path: Union[str, Path], revision: Optional[str] = None, cache_dir: Optional[str] = None) -> Path:
    local = Path(repo_or_path).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(repo_or_path), revision=revision, cache_dir=cache_dir, allow_patterns=list(ARTIFACT_FILES)))


def load_config(root: Path) -> SoproConfig:
    with open(root / CONFIG_FILE, "r", encoding="utf-8") as f:
        return SoproConfig.from_dict(json.load(f))


def load_weights(root: Path, name: str, device: torch.device) -> Dict[str, torch.Tensor]:
    return load_file(str(root / name), device=str(device))
