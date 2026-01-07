from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    CodebookEmbedding,
    RefXAttn,
    RMSNorm,
    SinusoidalPositionalEmbedding,
    SpeakerFiLM,
    SSMLiteBlock,
    TextEmbedding,
    TextXAttnBlock,
    Token2SV,
)
from .sampling import center_crop_tokens, repeated_tail
from .sampling import rf_ar as rf_ar_fn
from .sampling import rf_nar as rf_nar_fn
from .sampling import sample_token
from .tokenizer import TextTokenizer


class TextEncoder(nn.Module):
    def __init__(
        self,
        cfg: SoproTTSConfig,
        d_model: int,
        n_layers: int,
        tokenizer: TextTokenizer,
    ):
        super().__init__()
        self.tok = tokenizer
        self.embed = TextEmbedding(self.tok.vocab_size, d_model)
        self.layers = nn.ModuleList(
            [SSMLiteBlock(d_model, cfg.dropout, causal=False) for _ in range(n_layers)]
        )
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=cfg.max_text_len + 8)
        self.norm = RMSNorm(d_model)

    def forward(
        self, text_ids: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(text_ids)
        L = x.size(1)
        pos = self.pos(torch.arange(L, device=x.device))
        x = x + pos.unsqueeze(0)

        x = x * mask.unsqueeze(-1).float()
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
        return x, pooled


class ARRVQ1Generator(nn.Module):
    def __init__(self, cfg: SoproTTSConfig, d_model: int, vocab: int):
        super().__init__()
        ks = cfg.ar_kernel
        dils: List[int] = []
        while len(dils) < cfg.n_layers_ar:
            dils.extend(list(cfg.ar_dilation_cycle))
        dils = dils[: cfg.n_layers_ar]

        self.dils = tuple(int(d) for d in dils)
        self.blocks = nn.ModuleList(
            [
                SSMLiteBlock(
                    d_model, cfg.dropout, causal=True, kernel_size=ks, dilation=d
                )
                for d in self.dils
            ]
        )

        self.attn_freq = int(cfg.ar_text_attn_freq)
        self.x_attns = nn.ModuleList()
        for i in range(len(self.blocks)):
            if (i + 1) % self.attn_freq == 0:
                self.x_attns.append(
                    TextXAttnBlock(d_model, heads=4, dropout=cfg.dropout)
                )
            else:
                self.x_attns.append(nn.Identity())

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = ~text_mask if text_mask is not None else None

        if key_padding_mask is not None:
            bad_rows = key_padding_mask.all(dim=1)
            if bad_rows.any():
                key_padding_mask = key_padding_mask.clone()
                idx = torch.nonzero(bad_rows, as_tuple=False).squeeze(1)
                key_padding_mask[idx, 0] = False
                if text_emb is not None:
                    text_emb = text_emb.clone()
                    text_emb[idx, 0, :] = 0

        h = x
        for i, lyr in enumerate(self.blocks):
            h = lyr(h)
            if not isinstance(self.x_attns[i], nn.Identity) and text_emb is not None:
                h = self.x_attns[i](h, text_emb, key_padding_mask=key_padding_mask)

        h = self.norm(h)
        return self.head(h)


class StageRefiner(nn.Module):
    def __init__(self, cfg: SoproTTSConfig, D: int, num_heads: int, codebook_size: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SSMLiteBlock(D, cfg.dropout, causal=True) for _ in range(cfg.n_layers_nar)]
        )
        self.norm = RMSNorm(D)
        self.pre = nn.Linear(D, cfg.nar_head_dim)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.nar_head_dim, codebook_size) for _ in range(num_heads)]
        )
        self.mix = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward_hidden(
        self, cond_bt_d: torch.Tensor, prev_bt_d: torch.Tensor
    ) -> torch.Tensor:
        w = torch.softmax(self.mix, dim=0)
        x = w[0] * cond_bt_d + w[1] * prev_bt_d
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward_heads(self, h: torch.Tensor) -> List[torch.Tensor]:
        z = self.pre(h)
        return [head(z) for head in self.heads]


class StopHead(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.proj = nn.Linear(D, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)


class SoproTTSModel(nn.Module):
    def __init__(self, cfg: SoproTTSConfig, tokenizer: TextTokenizer):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.d_model)

        self.text_enc = TextEncoder(cfg, D, cfg.n_layers_text, tokenizer)
        self.frame_pos = SinusoidalPositionalEmbedding(D, max_len=cfg.pos_emb_max + 2)

        self.cb_embed = CodebookEmbedding(
            cfg.num_codebooks, cfg.codebook_size, D, use_bos=True
        )
        self.rvq1_bos_id = self.cb_embed.bos_id

        self.token2sv = Token2SV(
            cfg.num_codebooks,
            cfg.codebook_size,
            d=192,
            out_dim=cfg.sv_student_dim,
            dropout=cfg.dropout,
        )
        self.spk_film = SpeakerFiLM(D, sv_dim=cfg.sv_student_dim)
        self.cond_norm = RMSNorm(D)

        self.ar = ARRVQ1Generator(cfg, D, cfg.codebook_size)
        if cfg.ar_lookback > 0:
            self.ar_hist_w = nn.Parameter(torch.zeros(cfg.ar_lookback))

        def idxs(rng: Tuple[int, int]) -> List[int]:
            lo, hi = rng
            return list(range(lo - 1, hi))

        self.stage_indices: Dict[str, List[int]] = {
            "B": idxs(cfg.stage_B),
            "C": idxs(cfg.stage_C),
            "D": idxs(cfg.stage_D),
            "E": idxs(cfg.stage_E),
        }
        self.stages = nn.ModuleDict(
            {
                s: StageRefiner(cfg, D, len(self.stage_indices[s]), cfg.codebook_size)
                for s in ["B", "C", "D", "E"]
            }
        )

        self.stop_head = StopHead(D) if cfg.use_stop_head else None

        self.ref_enc_blocks = nn.ModuleList(
            [SSMLiteBlock(D, cfg.dropout, causal=False) for _ in range(2)]
        )
        self.ref_enc_norm = RMSNorm(D)
        self.ref_xattn_stack = RefXAttn(
            D, heads=cfg.ref_attn_heads, layers=3, dropout=cfg.dropout
        )

    def rf_ar(self) -> int:
        return rf_ar_fn(self.cfg.ar_kernel, self.ar.dils)

    def rf_nar(self) -> int:
        return rf_nar_fn(self.cfg.n_layers_nar, kernel_size=7, dilation=1)

    def _pool_time(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1 or x.size(1) < 2 * factor:
            return x
        return F.avg_pool1d(
            x.transpose(1, 2), kernel_size=factor, stride=factor
        ).transpose(1, 2)

    def _normalize_ref_mask(
        self, ref_mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        if ref_mask is None:
            return None
        mk = ref_mask.to(device).bool()
        if mk.ndim == 1:
            mk = mk.unsqueeze(0)
        return mk

    def _encode_reference_seq(self, ref_tokens: torch.Tensor) -> torch.Tensor:
        B, Tr, Q = ref_tokens.shape
        emb_sum = 0.0
        for q in range(Q):
            emb_sum = emb_sum + self.cb_embed.embed_tokens(
                ref_tokens[:, :, q], cb_index=q
            )
        x = emb_sum / float(Q)
        for b in self.ref_enc_blocks:
            x = b(x)
        return self.ref_enc_norm(x)

    def _single_pass_ref_xattn(
        self,
        cond_btd: torch.Tensor,
        ref_seq: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ref_seq_p = self._pool_time(ref_seq, 1)

        key_padding_mask = None
        if ref_mask is not None:
            mk_bool = ref_mask.bool()
            B, Tr = mk_bool.shape
            pooled_len = ref_seq_p.size(1)
            if pooled_len == Tr:
                key_padding_mask = ~mk_bool
            else:
                cut = pooled_len * 2
                mk2 = mk_bool[:, :cut].reshape(B, pooled_len, 2).any(dim=2)
                key_padding_mask = ~mk2

        return self.ref_xattn_stack(
            cond_btd, ref_seq_p, key_padding_mask=key_padding_mask
        )

    def _base_cond_at(
        self, t: int, txt_pool: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        pos = self.frame_pos(torch.tensor([t], device=device)).unsqueeze(0)
        return txt_pool[:, None, :] + pos

    def _ar_prev_from_seq(self, seq_1xT: torch.Tensor) -> torch.Tensor:
        K = int(self.cfg.ar_lookback)
        if K <= 0 or getattr(self, "ar_hist_w", None) is None:
            return self.cb_embed.embed_shift_by_k(seq_1xT, cb_index=0, k=1)

        ws = torch.softmax(self.ar_hist_w, dim=0)
        acc = 0.0
        k_max = min(K, int(seq_1xT.size(1)))
        for k in range(1, k_max + 1):
            acc = acc + ws[k - 1] * self.cb_embed.embed_shift_by_k(
                seq_1xT, cb_index=0, k=k
            )
        return acc

    @torch.no_grad()
    def prepare_conditioning(
        self,
        text_ids_1d: torch.Tensor,
        ref_tokens_tq: torch.Tensor,
        *,
        max_frames: int,
        device: torch.device,
        style_strength: float = 1.0,
        ref_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()

        text_ids = text_ids_1d.to(device)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool).unsqueeze(0)
        txt_seq, txt_pool = self.text_enc(text_ids.unsqueeze(0), text_mask)

        ref_btq = ref_tokens_tq.unsqueeze(0).to(device)

        sv_ref = self.token2sv(ref_btq, lengths=None)
        ref_seq = self._encode_reference_seq(ref_btq)
        ref_mask_btr = self._normalize_ref_mask(ref_mask, device)

        T = int(max_frames)
        if T <= 0:
            cond_all = torch.zeros(
                (1, 0, txt_pool.size(-1)), device=device, dtype=txt_pool.dtype
            )
        else:
            pos = self.frame_pos(torch.arange(T, device=device)).unsqueeze(0)
            base_all = txt_pool[:, None, :] + pos
            base_all = self.spk_film(base_all, sv_ref, strength=float(style_strength))

            if chunk_size is None or int(chunk_size) >= T:
                out = self._single_pass_ref_xattn(
                    base_all, ref_seq, ref_mask=ref_mask_btr
                )
                cond_all = self.cond_norm(out)
            else:
                cs = int(chunk_size)
                chunks: List[torch.Tensor] = []
                for s in range(0, T, cs):
                    e = min(T, s + cs)
                    q = base_all[:, s:e, :]
                    out = self._single_pass_ref_xattn(q, ref_seq, ref_mask=ref_mask_btr)
                    chunks.append(self.cond_norm(out))
                cond_all = torch.cat(chunks, dim=1)

        return {
            "txt_seq": txt_seq,
            "text_mask": text_mask,
            "cond_all": cond_all,
            "ref_btq": ref_btq,
            "txt_pool": txt_pool,
            "sv_ref": sv_ref,
            "ref_seq": ref_seq,
            "ref_mask": (
                ref_mask_btr
                if ref_mask_btr is not None
                else torch.empty(0, device=device, dtype=torch.bool)
            ),
            "style_strength": torch.tensor(float(style_strength), device=device),
        }

    @torch.no_grad()
    def prepare_conditioning_lazy(
        self,
        text_ids_1d: torch.Tensor,
        ref_tokens_tq: torch.Tensor,
        *,
        max_frames: int,
        device: torch.device,
        style_strength: float = 1.0,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()

        text_ids = text_ids_1d.to(device)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool).unsqueeze(0)
        txt_seq, txt_pool = self.text_enc(text_ids.unsqueeze(0), text_mask)

        ref_btq = ref_tokens_tq.unsqueeze(0).to(device)
        sv_ref = self.token2sv(ref_btq, lengths=None)
        ref_seq = self._encode_reference_seq(ref_btq)
        ref_mask_btr = self._normalize_ref_mask(ref_mask, device)

        D = int(txt_pool.size(-1))
        cond_all = torch.zeros((1, 0, D), device=device, dtype=txt_pool.dtype)

        return {
            "txt_seq": txt_seq,
            "text_mask": text_mask,
            "cond_all": cond_all,
            "ref_btq": ref_btq,
            "txt_pool": txt_pool,
            "sv_ref": sv_ref,
            "ref_seq": ref_seq,
            "ref_mask": (
                ref_mask_btr
                if ref_mask_btr is not None
                else torch.empty(0, device=device, dtype=torch.bool)
            ),
            "style_strength": torch.tensor(float(style_strength), device=device),
            "max_frames": torch.tensor(int(max_frames), device=device),
        }

    @torch.no_grad()
    def ensure_cond_upto(
        self,
        prep: Dict[str, torch.Tensor],
        t_inclusive: int,
        *,
        chunk_size: int = 64,
    ) -> None:
        if t_inclusive < 0:
            return

        cond_all = prep.get("cond_all", None)
        if cond_all is None:
            raise KeyError("prep dict missing 'cond_all'.")

        have = int(cond_all.size(1))
        need_min = int(t_inclusive) + 1
        if have >= need_min:
            return

        if "txt_pool" not in prep or "sv_ref" not in prep or "ref_seq" not in prep:
            raise RuntimeError(
                "Lazy conditioning requested but prep lacks txt_pool/sv_ref/ref_seq. "
                "Use prepare_conditioning_lazy() or prepare_conditioning(..., chunk_size=...)."
            )

        device = cond_all.device
        txt_pool = prep["txt_pool"]
        sv_ref = prep["sv_ref"]
        ref_seq = prep["ref_seq"]

        style_strength = float(
            prep.get(
                "style_strength", torch.tensor(self.cfg.style_strength, device=device)
            ).item()
        )

        ref_mask = prep.get("ref_mask", None)
        if ref_mask is not None and ref_mask.numel() == 0:
            ref_mask = None

        max_frames = prep.get("max_frames", None)
        maxT = int(max_frames.item()) if max_frames is not None else None

        cs = max(1, int(chunk_size))

        need = ((need_min + cs - 1) // cs) * cs
        if maxT is not None:
            need = min(need, maxT)

        if have >= need:
            return

        new_chunks: List[torch.Tensor] = []
        for s in range(have, need, cs):
            e = min(need, s + cs)
            pos = self.frame_pos(torch.arange(s, e, device=device)).unsqueeze(0)
            base = txt_pool[:, None, :] + pos
            base = self.spk_film(base, sv_ref, strength=style_strength)
            out = self._single_pass_ref_xattn(base, ref_seq, ref_mask=ref_mask)
            new_chunks.append(self.cond_norm(out))

        prep["cond_all"] = torch.cat([cond_all] + new_chunks, dim=1)

    def build_ar_prefix(
        self,
        ref_btq: torch.Tensor,
        device: torch.device,
        prefix_sec_fixed: Optional[float],
        use_prefix: bool,
    ) -> torch.Tensor:
        if not use_prefix or ref_btq.size(1) == 0:
            return torch.zeros(1, 0, dtype=torch.long, device=device)

        avail = int(ref_btq.size(1))
        fps = float(self.cfg.mimi_fps)

        if prefix_sec_fixed is not None and prefix_sec_fixed > 0:
            P = min(avail, int(round(prefix_sec_fixed * fps)))
        else:
            P = min(avail, max(1, int(round(self.cfg.preprompt_sec_max * fps))))

        if P <= 0:
            return torch.zeros(1, 0, dtype=torch.long, device=device)
        return ref_btq[:, :P, 0].contiguous()

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
        use_prefix: bool = True,
        prefix_sec_fixed: Optional[float] = None,
        cond_chunk_size: Optional[int] = None,
        use_stop_head: Optional[bool] = None,
        stop_patience: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        min_gen_frames: Optional[int] = None,
    ) -> Iterator[Tuple[int, int, Optional[float]]]:
        device = prep["cond_all"].device
        cond_all = prep["cond_all"]
        txt_seq = prep["txt_seq"]
        text_mask = prep["text_mask"]
        ref_btq = prep["ref_btq"]

        R_AR = self.rf_ar()

        stop_head = self.stop_head
        if use_stop_head is not None:
            if not bool(use_stop_head):
                stop_head = None

        eff_stop_patience = int(
            stop_patience if stop_patience is not None else self.cfg.stop_patience
        )
        eff_stop_threshold = float(
            stop_threshold if stop_threshold is not None else self.cfg.stop_threshold
        )
        eff_min_gen_frames = int(
            min_gen_frames if min_gen_frames is not None else self.cfg.min_gen_frames
        )

        A_prefix = self.build_ar_prefix(
            ref_btq, device, prefix_sec_fixed, use_prefix=use_prefix
        )
        P = int(A_prefix.size(1))

        ctx_ids = torch.zeros(
            (1, P + int(max_frames) + 1), dtype=torch.long, device=device
        )
        if P > 0:
            ctx_ids[:, :P] = A_prefix

        hist_A: List[int] = []
        loop_streak_count = 0
        stop_streak_count = 0
        last_a: Optional[int] = None

        gen_len = 0

        for t in range(int(max_frames)):
            if prep["cond_all"].size(1) < (t + 1):
                self.ensure_cond_upto(prep, t, chunk_size=int(cond_chunk_size or 64))
            cond_all = prep["cond_all"]

            L_ar = min(t + 1, R_AR)
            s_ar = t + 1 - L_ar
            cond_win_ar = cond_all[:, s_ar : t + 1, :]

            total_len = P + gen_len + 1
            A_ctx_full = ctx_ids[:, :total_len]

            prev_ctx_full = self._ar_prev_from_seq(A_ctx_full)
            prev_ctx_win = prev_ctx_full[:, -L_ar:, :]

            cur_top_p, cur_temp = top_p, temperature
            if anti_loop:
                if repeated_tail(hist_A, max_n=16):
                    cur_top_p, cur_temp = recovery_top_p, recovery_temp
                elif last_a is not None and loop_streak_count >= loop_streak:
                    cur_top_p, cur_temp = recovery_top_p, recovery_temp

            ar_logits_win = self.ar(
                cond_win_ar + prev_ctx_win, text_emb=txt_seq, text_mask=text_mask
            )
            ar_logits_t = ar_logits_win[:, -1:, :]

            rvq1_id = sample_token(
                ar_logits_t,
                history=hist_A,
                top_p=cur_top_p,
                temperature=cur_temp,
                top_k=50,
                repetition_penalty=1.1,
            )

            ctx_ids[0, P + gen_len] = int(rvq1_id)
            gen_len += 1

            hist_A.append(int(rvq1_id))
            loop_streak_count = (
                (loop_streak_count + 1)
                if (last_a is not None and rvq1_id == last_a)
                else 0
            )
            last_a = int(rvq1_id)

            p_stop: Optional[float] = None
            if stop_head is not None:
                A_now = torch.tensor([[rvq1_id]], device=device, dtype=torch.long)
                stop_inp = (
                    cond_all[:, t : t + 1, :]
                    + self.cb_embed.embed_tokens(A_now, cb_index=0).detach()
                )
                stop_logits = stop_head(stop_inp)
                p_stop = float(torch.sigmoid(stop_logits).item())

                if t + 1 >= eff_min_gen_frames and p_stop > eff_stop_threshold:
                    stop_streak_count += 1
                else:
                    stop_streak_count = 0

            yield t, int(rvq1_id), p_stop

            if stop_head is not None and stop_streak_count >= eff_stop_patience:
                break

    @torch.no_grad()
    def nar_refine(
        self, cond_seq: torch.Tensor, tokens_A_1xT: torch.Tensor
    ) -> torch.Tensor:
        preds_all: List[torch.Tensor] = [tokens_A_1xT.unsqueeze(-1)]
        prev_tokens_list: List[torch.Tensor] = [tokens_A_1xT.unsqueeze(-1)]
        prev_cb_list: List[List[int]] = [[0]]

        for stage_name in ["B", "C", "D", "E"]:
            idxs = self.stage_indices[stage_name]
            prev_tokens_cat = torch.cat(prev_tokens_list, dim=-1)
            prev_cbs_cat = sum(prev_cb_list, [])
            prev_emb_sum = self.cb_embed.sum_embed_subset(prev_tokens_cat, prev_cbs_cat)

            h = self.stages[stage_name].forward_hidden(cond_seq, prev_emb_sum)
            logits_list = self.stages[stage_name].forward_heads(h)
            preds = torch.stack([x.argmax(dim=-1) for x in logits_list], dim=-1)

            preds_all.append(preds)
            prev_tokens_list.append(preds)
            prev_cb_list.append(idxs)

        tokens_btq = torch.cat(preds_all, dim=-1)
        return tokens_btq

    @torch.no_grad()
    def generate_tokens(
        self,
        text_ids_1d: torch.Tensor,
        ref_tokens_tq: torch.Tensor,
        *,
        max_frames: int,
        device: torch.device,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        use_prefix: bool = True,
        prefix_sec_fixed: Optional[float] = None,
        style_strength: float = 1.0,
        use_stop_head: Optional[bool] = None,
        stop_patience: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        min_gen_frames: Optional[int] = None,
    ) -> torch.Tensor:
        prep = self.prepare_conditioning(
            text_ids_1d,
            ref_tokens_tq,
            max_frames=max_frames,
            device=device,
            style_strength=style_strength,
        )

        hist_A: List[int] = []
        for _t, rvq1, _p_stop in self.ar_stream(
            prep,
            max_frames=max_frames,
            top_p=top_p,
            temperature=temperature,
            anti_loop=anti_loop,
            use_prefix=use_prefix,
            prefix_sec_fixed=prefix_sec_fixed,
            use_stop_head=use_stop_head,
            stop_patience=stop_patience,
            stop_threshold=stop_threshold,
            min_gen_frames=min_gen_frames,
        ):
            hist_A.append(rvq1)

        T = len(hist_A)
        if T == 0:
            return torch.zeros(
                0, self.cfg.num_codebooks, dtype=torch.long, device=device
            )

        tokens_A = torch.tensor(hist_A, device=device, dtype=torch.long).unsqueeze(0)
        cond_seq = prep["cond_all"][:, :T, :]
        tokens_btq_1xTQ = self.nar_refine(cond_seq, tokens_A)
        return tokens_btq_1xTQ.squeeze(0)


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

        model.load_state_dict(state)

        codec = MimiCodec(num_quantizers=cfg.num_codebooks, device=device)

        return cls(
            model=model, cfg=cfg, tokenizer=tokenizer, codec=codec, device=device
        )

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def encode_reference(
        self,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        ref_seconds: Optional[float] = None,
    ) -> torch.Tensor:
        if (ref_tokens_tq is None) and (ref_audio_path is None):
            raise RuntimeError(
                "SoproTTS requires a reference. Provide ref_audio_path=... or ref_tokens_tq=..."
            )
        if (ref_tokens_tq is not None) and (ref_audio_path is not None):
            raise RuntimeError(
                "Provide only one of ref_audio_path or ref_tokens_tq (not both)."
            )

        if ref_seconds is None:
            ref_seconds = float(self.cfg.ref_seconds_max)

        if ref_tokens_tq is not None:
            ref = ref_tokens_tq.to(self.device).long()
            if ref_seconds > 0:
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

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        *,
        ref_audio_path: Optional[str] = None,
        ref_tokens_tq: Optional[torch.Tensor] = None,
        max_frames: int = 400,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        use_prefix: bool = True,
        prefix_sec_fixed: Optional[float] = None,
        style_strength: Optional[float] = None,
        ref_seconds: Optional[float] = None,
        use_stop_head: Optional[bool] = None,
        stop_patience: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        min_gen_frames: Optional[int] = None,
    ) -> torch.Tensor:
        text_ids = self.encode_text(text)
        ref = self.encode_reference(
            ref_audio_path=ref_audio_path,
            ref_tokens_tq=ref_tokens_tq,
            ref_seconds=ref_seconds,
        )

        tokens_tq = self.model.generate_tokens(
            text_ids,
            ref,
            max_frames=max_frames,
            device=self.device,
            top_p=top_p,
            temperature=temperature,
            anti_loop=anti_loop,
            use_prefix=use_prefix,
            prefix_sec_fixed=prefix_sec_fixed,
            style_strength=float(
                style_strength
                if style_strength is not None
                else self.cfg.style_strength
            ),
            use_stop_head=use_stop_head,
            stop_patience=stop_patience,
            stop_threshold=stop_threshold,
            min_gen_frames=min_gen_frames,
        )

        wav = self.codec.decode_full(tokens_tq)
        return wav

    def stream(self, text: str, **kwargs) -> Iterator[torch.Tensor]:
        from .streaming import stream as _stream

        return _stream(self, text, **kwargs)

    def save_wav(self, path: str, wav_1xT: torch.Tensor) -> None:
        save_audio(path, wav_1xT, sr=TARGET_SR)
