from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from sopro.config import SoproTTSConfig
from sopro.hub import (
    download_repo,
    load_cfg_from_safetensors,
    load_state_dict_from_safetensors,
)

from .audio import save_audio
from .codec.mimi import MimiCodec
from .constants import TARGET_SR
from .nn import (
    ARRVQ1Generator,
    CodebookEmbedding,
    RefXAttnStack,
    RMSNorm,
    SinusoidalPositionalEmbedding,
    SpeakerFiLM,
    SSMLiteBlock,
    TextEncoder,
    Token2SV,
)
from .nn.nar import NARSinglePass
from .sampling import repeated_tail
from .sampling import rf_ar as rf_ar_fn
from .sampling import rf_nar as rf_nar_fn
from .sampling import sample_token
from .tokenizer import TextTokenizer


def _stage_range_to_indices(stage_rng: Tuple[int, int], Q: int) -> List[int]:
    lo, hi = int(stage_rng[0]), int(stage_rng[1])
    idxs = list(range(lo - 1, hi))
    return [i for i in idxs if 1 <= i < Q]


@dataclass
class PreparedReference:
    ref_tokens_btq: torch.Tensor
    sv_ref: torch.Tensor
    ref_seq: torch.Tensor
    ref_kv_caches: List[Dict[str, torch.Tensor]]


class SoproTTSModel(nn.Module):
    def __init__(self, cfg: SoproTTSConfig, tokenizer: TextTokenizer):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.d_model)

        self.eos_id = int(cfg.codebook_size)

        self.text_enc = TextEncoder(cfg, D, int(cfg.n_layers_text), tokenizer)
        self.frame_pos = SinusoidalPositionalEmbedding(
            D, max_len=int(cfg.pos_emb_max) + 8
        )

        self.cb_embed = CodebookEmbedding(
            cfg.num_codebooks, cfg.codebook_size, D, use_bos=True
        )

        self.nar_prev_cb_weights = nn.Parameter(
            torch.zeros(cfg.num_codebooks, dtype=torch.float32)
        )

        self.token2sv = Token2SV(
            cfg.num_codebooks,
            cfg.codebook_size,
            d=192,
            out_dim=int(cfg.sv_student_dim),
            dropout=cfg.dropout,
        )
        self.spk_film = SpeakerFiLM(D, sv_dim=int(cfg.sv_student_dim))

        self.ar = ARRVQ1Generator(cfg, D, int(cfg.codebook_size) + 1)

        Q = int(cfg.num_codebooks)
        self.stage_indices: Dict[str, List[int]] = {
            "B": _stage_range_to_indices(cfg.stage_B, Q),
            "C": _stage_range_to_indices(cfg.stage_C, Q),
            "D": _stage_range_to_indices(cfg.stage_D, Q),
            "E": _stage_range_to_indices(cfg.stage_E, Q),
        }
        self.stage_order = [
            s for s in ["B", "C", "D", "E"] if len(self.stage_indices[s]) > 0
        ]

        self.nar = NARSinglePass(cfg, D, stage_specs=self.stage_indices)

        self.cond_norm = RMSNorm(D)

        ref_enc_layers = int(getattr(cfg, "ref_enc_layers", 2))
        self.ref_enc_blocks = nn.ModuleList(
            [SSMLiteBlock(D, cfg.dropout, causal=False) for _ in range(ref_enc_layers)]
        )
        self.ref_enc_norm = RMSNorm(D)

        self.ref_xattn = RefXAttnStack(
            D,
            heads=cfg.ref_xattn_heads,
            layers=cfg.ref_xattn_layers,
            gmax=cfg.ref_xattn_gmax,
        )

        self.register_buffer(
            "ref_cb_weights",
            torch.linspace(1.0, 0.1, int(cfg.num_codebooks)),
            persistent=True,
        )

    def rf_ar(self) -> int:
        return rf_ar_fn(
            int(self.cfg.ar_kernel),
            getattr(self.ar, "dils", tuple(int(x) for x in self.cfg.ar_dilation_cycle)),
        )

    def rf_nar(self) -> int:
        cycle = tuple(int(x) for x in self.cfg.nar_dilation_cycle) or (1,)
        dils: List[int] = []
        while len(dils) < int(self.cfg.n_layers_nar):
            dils.extend(list(cycle))
        dils = dils[: int(self.cfg.n_layers_nar)]
        return rf_nar_fn(int(self.cfg.nar_kernel_size), tuple(dils))

    @torch.no_grad()
    def _encode_reference_seq(self, ref_tokens_btq: torch.Tensor) -> torch.Tensor:
        B, Tr, Q = ref_tokens_btq.shape

        w = torch.softmax(self.ref_cb_weights.float(), dim=0).to(
            device=ref_tokens_btq.device
        )

        x = 0.0
        for q in range(Q):
            e = self.cb_embed.embed_tokens(ref_tokens_btq[:, :, q], cb_index=q)
            x = x + w[q].to(e.dtype) * e

        for b in self.ref_enc_blocks:
            x = b(x)

        return self.ref_enc_norm(x)

    @torch.no_grad()
    def prepare_reference(
        self, ref_tokens_tq: torch.Tensor, *, device: torch.device
    ) -> PreparedReference:
        ref_tokens_btq = ref_tokens_tq.unsqueeze(0).to(device=device, dtype=torch.long)
        Tr = int(ref_tokens_btq.size(1))

        lengths = torch.tensor([Tr], device=device, dtype=torch.long)
        sv_ref = self.token2sv(ref_tokens_btq, lengths=lengths)

        ref_seq = self._encode_reference_seq(ref_tokens_btq)

        ref_kv_caches = self.ref_xattn.build_kv_caches(ref_seq, key_padding_mask=None)

        return PreparedReference(
            ref_tokens_btq=ref_tokens_btq,
            sv_ref=sv_ref,
            ref_seq=ref_seq,
            ref_kv_caches=ref_kv_caches,
        )

    @torch.no_grad()
    def prepare_conditioning(
        self,
        text_ids_1d: torch.Tensor,
        ref: PreparedReference,
        *,
        max_frames: int,
        device: torch.device,
        style_strength: float = 1.2,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        sv_ref = ref.sv_ref.to(device)

        text_ids = text_ids_1d.to(device)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool).unsqueeze(0)
        txt_seq, txt_pool = self.text_enc(text_ids.unsqueeze(0), text_mask)

        if sv_ref is not None:
            if sv_ref.dim() == 1:
                sv_ref = sv_ref.unsqueeze(0)
            sv_ref = sv_ref.to(device)
        else:
            ref_btq = ref_tokens_tq.unsqueeze(0).to(device)
            ref_len = torch.tensor(
                [int(ref_btq.size(1))], device=device, dtype=torch.long
            )
            sv_ref = self.token2sv(ref_btq, lengths=ref_len)

        Tar = int(max_frames) + 1
        pos = self.frame_pos(torch.arange(Tar, device=device)).unsqueeze(0)
        base_ar = txt_pool[:, None, :] + pos
        cond_ar = self.spk_film(base_ar, sv_ref, strength=float(style_strength))

        cond_ar, _ = self.ref_xattn(
            cond_ar, kv_caches=ref.ref_kv_caches, use_cache=True
        )
        cond_ar = self.cond_norm(cond_ar)

        return {
            "txt_seq": txt_seq,
            "text_mask": text_mask,
            "txt_pool": txt_pool,
            "sv_ref": sv_ref,
            "cond_ar": cond_ar,
        }

    @torch.no_grad()
    def ar_stream(
        self,
        prep: Dict[str, torch.Tensor],
        *,
        max_frames: int,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        loop_streak: int = 8,
        recovery_top_p: float = 0.85,
        recovery_temp: float = 1.2,
        min_gen_frames: Optional[int] = None,
    ) -> Iterator[Tuple[int, int, bool]]:
        device = prep["cond_ar"].device
        cond_ar = prep["cond_ar"]
        txt_seq = prep["txt_seq"]
        text_mask = prep["text_mask"]

        eos_id = int(self.eos_id)
        eff_min_gen = int(
            min_gen_frames if min_gen_frames is not None else self.cfg.min_gen_frames
        )

        max_steps = int(max_frames) + 1
        ctx_ids = torch.zeros((1, max_steps), dtype=torch.long, device=device)

        ar_state = self.ar.init_stream_state(
            batch_size=1,
            device=device,
            dtype=cond_ar.dtype,
            text_emb=txt_seq,
            text_mask=text_mask,
        )

        hist: List[int] = []
        loop_streak_count = 0
        last_a: Optional[int] = None

        if self.cb_embed.bos_id is None:
            raise RuntimeError(
                "CodebookEmbedding.use_bos must be True for streaming AR cache"
            )
        bos_idx = torch.full(
            (1, 1), int(self.cb_embed.bos_id), device=device, dtype=torch.long
        )

        for t in range(max_steps):
            if t == 0:
                prev_emb = self.cb_embed.emb(bos_idx)
            else:
                prev_tok = ctx_ids[:, t - 1 : t]
                prev_emb = self.cb_embed.embed_tokens(prev_tok, cb_index=0)

            x_t = cond_ar[:, t : t + 1, :] + prev_emb

            cur_top_p, cur_temp = top_p, temperature
            if anti_loop:
                if repeated_tail(hist, max_n=16):
                    cur_top_p, cur_temp = recovery_top_p, recovery_temp
                elif last_a is not None and loop_streak_count >= loop_streak:
                    cur_top_p, cur_temp = recovery_top_p, recovery_temp

            logits_t, ar_state = self.ar.step(
                x_t, ar_state, text_emb=txt_seq, text_mask=text_mask
            )
            tok = sample_token(
                logits_t,
                history=hist,
                top_p=cur_top_p,
                temperature=cur_temp,
                top_k=50,
                repetition_penalty=1.1,
            )

            ctx_ids[0, t] = int(tok)
            hist.append(int(tok))

            loop_streak_count = (
                (loop_streak_count + 1) if (last_a is not None and tok == last_a) else 0
            )
            last_a = int(tok)

            is_eos = int(tok) == eos_id
            yield t, int(tok), bool(is_eos)

            if is_eos and (t + 1) >= eff_min_gen:
                break

    @torch.no_grad()
    def nar_refine(
        self, cond_seq: torch.Tensor, rvq1_1xT: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = cond_seq.shape
        Q = int(self.cfg.num_codebooks)

        out_btq = torch.zeros((B, T, Q), device=cond_seq.device, dtype=torch.long)
        out_btq[:, :, 0] = rvq1_1xT

        prev_tokens_list: List[torch.Tensor] = [rvq1_1xT.unsqueeze(-1)]
        prev_cb_list: List[List[int]] = [[0]]

        for stage in self.stage_order:
            idxs = self.stage_indices[stage]
            if len(idxs) == 0:
                continue

            prev_tokens_cat = torch.cat(prev_tokens_list, dim=-1)
            prev_cbs_cat = sum(prev_cb_list, [])

            prev_emb_sum = self.cb_embed.sum_embed_subset(
                prev_tokens_cat,
                prev_cbs_cat,
                keep_mask=None,
                cb_weights=self.nar_prev_cb_weights,
            )

            logits_list = self.nar.forward_stage(stage, cond_seq, prev_emb_sum)
            if len(logits_list) == 0:
                continue

            preds = torch.stack([lg.argmax(dim=-1) for lg in logits_list], dim=-1)

            for k, cb in enumerate(idxs):
                out_btq[:, :, cb] = preds[:, :, k]

            prev_tokens_list.append(preds.detach())
            prev_cb_list.append(idxs)

        return out_btq

    @torch.no_grad()
    def generate_tokens(
        self,
        text_ids_1d: torch.Tensor,
        ref: PreparedReference,
        *,
        max_frames: int,
        device: torch.device,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        style_strength: float = 1.2,
        min_gen_frames: Optional[int] = None,
    ) -> torch.Tensor:
        prep = self.prepare_conditioning(
            text_ids_1d,
            ref,
            max_frames=max_frames,
            device=device,
            style_strength=style_strength,
        )

        eos_id = int(self.eos_id)
        hist: List[int] = []
        for _t, tok, is_eos in self.ar_stream(
            prep,
            max_frames=max_frames,
            top_p=top_p,
            temperature=temperature,
            anti_loop=anti_loop,
            min_gen_frames=min_gen_frames,
        ):
            hist.append(tok)
            if is_eos:
                break

        Tfull = len(hist)
        cut = Tfull
        for i, v in enumerate(hist):
            if int(v) == eos_id:
                cut = i
                break

        T = int(cut)
        if T <= 0:
            return torch.zeros(
                (0, int(self.cfg.num_codebooks)), dtype=torch.long, device=device
            )

        rvq1 = torch.tensor(hist[:T], device=device, dtype=torch.long).unsqueeze(0)
        cond_seq = prep["cond_ar"][:, :T, :]
        tokens_1xTQ = self.nar_refine(cond_seq, rvq1)
        return tokens_1xTQ.squeeze(0)


class SoproTTS:
    def __init__(
        self,
        model: SoproTTSModel,
        cfg: SoproTTSConfig,
        tokenizer: TextTokenizer,
        codec: MimiCodec,
        device: str,
    ):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.codec = codec
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "SoproTTS":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(device)

        local_dir = download_repo(
            repo_id, revision=revision, cache_dir=cache_dir, token=token
        )

        model_path = os.path.join(local_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected {model_path} in repo snapshot.")

        cfg = load_cfg_from_safetensors(model_path)
        tokenizer = TextTokenizer(model_name=local_dir)

        model = SoproTTSModel(cfg, tokenizer).to(dev).eval()
        state = load_state_dict_from_safetensors(model_path)

        model.load_state_dict(state, strict=False)

        codec = MimiCodec(num_quantizers=cfg.num_codebooks, device=device)
        return cls(
            model=model, cfg=cfg, tokenizer=tokenizer, codec=codec, device=device
        )

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def encode_speaker(
        self,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        ref_seconds: Optional[float] = None,
    ) -> torch.Tensor:
        ref = self.encode_reference(
            ref_audio_path=ref_audio_path,
            ref_tokens_tq=ref_tokens_tq,
            ref_seconds=ref_seconds,
        )
        ref_btq = ref.unsqueeze(0)
        lengths = torch.tensor(
            [int(ref_btq.size(1))], device=self.device, dtype=torch.long
        )
        sv = self.model.token2sv(ref_btq, lengths=lengths)
        return sv.squeeze(0).detach()

    def encode_reference(
        self,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        ref_seconds: Optional[float] = None,
    ) -> torch.Tensor:
        from .sampling import center_crop_tokens

        if (ref_tokens_tq is None) and (ref_audio_path is None):
            raise RuntimeError(
                "SoproTTS requires a reference. Provide ref_audio_path=... or ref_tokens_tq=..."
            )
        if (ref_tokens_tq is not None) and (ref_audio_path is not None):
            raise RuntimeError(
                "Provide only one of ref_audio_path or ref_tokens_tq (not both)."
            )

        if ref_seconds is None:
            ref_seconds = 12.0

        if ref_tokens_tq is not None:
            ref = ref_tokens_tq.to(self.device).long()
            if ref_seconds and ref_seconds > 0:
                fps = float(self.cfg.mimi_fps)
                win = max(1, int(round(ref_seconds * fps)))
                ref = center_crop_tokens(ref, win)
            return ref

        crop_seconds = (
            ref_seconds if (ref_seconds is not None and ref_seconds > 0) else None
        )
        ref = (
            self.codec.encode_file(ref_audio_path, crop_seconds=crop_seconds)
            .to(self.device)
            .long()
        )
        return ref

    @torch.inference_mode()
    def prepare_reference(
        self,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        ref_seconds: Optional[float] = None,
    ) -> PreparedReference:
        tokens_tq = self.encode_reference(
            ref_audio_path=ref_audio_path,
            ref_tokens_tq=ref_tokens_tq,
            ref_seconds=ref_seconds,
        )
        return self.model.prepare_reference(tokens_tq, device=self.device)

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        *,
        ref: Optional[PreparedReference] = None,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        max_frames: int = 400,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        style_strength: Optional[float] = None,
        ref_seconds: Optional[float] = None,
        min_gen_frames: Optional[int] = None,
    ) -> torch.Tensor:
        text_ids = self.encode_text(text)

        if ref is None:
            ref = self.prepare_reference(
                ref_audio_path=ref_audio_path,
                ref_tokens_tq=ref_tokens_tq,
                ref_seconds=ref_seconds,
            )

        text_ids = self.encode_text(text)

        tokens_tq = self.model.generate_tokens(
            text_ids,
            ref=ref,
            max_frames=max_frames,
            device=self.device,
            top_p=top_p,
            temperature=temperature,
            anti_loop=anti_loop,
            style_strength=float(
                style_strength
                if style_strength is not None
                else self.cfg.style_strength
            ),
            min_gen_frames=min_gen_frames,
        )

        wav = self.codec.decode_full(tokens_tq)
        return wav

    def stream(self, text: str, **kwargs) -> Iterator[torch.Tensor]:
        from .streaming import stream as _stream

        return _stream(self, text, **kwargs)

    def save_wav(self, path: str, wav_1xT: torch.Tensor) -> None:
        save_audio(path, wav_1xT, sr=TARGET_SR)
