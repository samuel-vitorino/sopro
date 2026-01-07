from __future__ import annotations

import json
import struct
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from sopro.config import SoproTTSConfig


def download_repo(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )


def _read_safetensors_metadata(path: str) -> Dict[str, str]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    meta = header.get("__metadata__", {}) or {}
    return {str(k): str(v) for k, v in meta.items()}


def load_cfg_from_safetensors(path: str) -> SoproTTSConfig:
    meta = _read_safetensors_metadata(path)
    if "cfg" not in meta:
        raise RuntimeError(f"No 'cfg' metadata found in {path}.")

    cfg_dict = json.loads(meta["cfg"])
    init: Dict[str, Any] = {}
    for k in SoproTTSConfig.__annotations__.keys():
        if k in cfg_dict:
            init[k] = cfg_dict[k]

    cfg = SoproTTSConfig(**init)
    return cfg


def load_state_dict_from_safetensors(path: str) -> Dict[str, torch.Tensor]:
    return load_file(path)
