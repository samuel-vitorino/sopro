from __future__ import annotations

from typing import List

try:
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    hf_logging = None
    AutoTokenizer = None


class TextTokenizer:
    def __init__(self, model_name: str, add_bos_eos: bool = True):
        if AutoTokenizer is None:
            raise ImportError("pip install transformers sentencepiece")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.add_bos_eos = add_bos_eos

        if self.tok.pad_token_id is None:
            self.tok.add_special_tokens({"pad_token": "<|pad|>"})

        self.pad_id = int(self.tok.pad_token_id)
        self.bos_id = (
            int(self.tok.bos_token_id) if self.tok.bos_token_id is not None else None
        )
        self.eos_id = (
            int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None
        )
        self.vocab_size = int(self.tok.vocab_size + len(self.tok.get_added_vocab()))

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if self.add_bos_eos and self.bos_id is not None and self.eos_id is not None:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids
