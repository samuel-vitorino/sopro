from __future__ import annotations

import argparse
import time

import torch
from tqdm.auto import tqdm

from .audio import save_audio
from .constants import TARGET_SR
from .model import SoproTTS


def main() -> None:
    ap = argparse.ArgumentParser(description="SoproTTS cli inference")

    ap.add_argument("--repo_id", type=str, default="samuel-vitorino/sopro")
    ap.add_argument(
        "--revision", type=str, default=None, help="Optional git revision/branch/tag"
    )
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache dir")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token")

    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--ref_audio", type=str, default=None)
    ap.add_argument("--ref_tokens", type=str, default=None, help="Path to .npy [T,Q]")
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--max_frames", type=int, default=400)

    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.05)
    ap.add_argument("--no_anti_loop", action="store_true")

    ap.add_argument("--no_prefix", action="store_true")
    ap.add_argument("--prefix_sec", type=float, default=None)
    ap.add_argument("--style_strength", type=float, default=None)
    ap.add_argument("--ref_seconds", type=float, default=None)

    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling")

    ap.add_argument(
        "--no_stop_head", action="store_true", help="Disable stop head early stopping"
    )
    ap.add_argument(
        "--stop_patience", type=int, default=None, help="Override cfg.stop_patience"
    )
    ap.add_argument(
        "--stop_threshold", type=float, default=None, help="Override cfg.stop_threshold"
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cuda if available else cpu)",
    )

    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device or default_device

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Error: --device cuda requested but CUDA is not available.")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("Error: --device mps requested but MPS is not available.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t0 = time.perf_counter()
    tts = SoproTTS.from_pretrained(
        args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        device=device,
    )
    t1 = time.perf_counter()
    if not args.quiet:
        print(f"[Load] {t1 - t0:.2f}s")

    cfg = tts.cfg

    ref_tokens_tq = None
    if args.ref_tokens is not None:
        import numpy as np

        arr = np.load(args.ref_tokens)
        ref_tokens_tq = torch.from_numpy(arr).long()

    text_ids = tts.encode_text(args.text)
    ref = tts.encode_reference(
        ref_audio_path=args.ref_audio,
        ref_tokens_tq=ref_tokens_tq,
        ref_seconds=args.ref_seconds,
    )

    prep = tts.model.prepare_conditioning(
        text_ids,
        ref,
        max_frames=args.max_frames,
        device=tts.device,
        style_strength=float(
            args.style_strength
            if args.style_strength is not None
            else cfg.style_strength
        ),
    )

    t_start = time.perf_counter()

    hist_A: list[int] = []
    pbar = tqdm(
        total=args.max_frames,
        desc="AR sampling",
        unit="frame",
        disable=args.quiet,
    )

    for _t, rvq1, p_stop in tts.model.ar_stream(
        prep,
        max_frames=args.max_frames,
        top_p=args.top_p,
        temperature=args.temperature,
        anti_loop=(not args.no_anti_loop),
        use_prefix=(not args.no_prefix),
        prefix_sec_fixed=args.prefix_sec,
        use_stop_head=(False if args.no_stop_head else None),
        stop_patience=args.stop_patience,
        stop_threshold=args.stop_threshold,
    ):
        hist_A.append(int(rvq1))
        pbar.update(1)
        if p_stop is None:
            pbar.set_postfix(p_stop="off")
        else:
            pbar.set_postfix(p_stop=f"{float(p_stop):.2f}")

    pbar.n = len(hist_A)
    pbar.close()

    t_after_sampling = time.perf_counter()

    T = len(hist_A)
    if T == 0:
        save_audio(args.out, torch.zeros(1, 0), sr=TARGET_SR)
        t_end = time.perf_counter()
        if not args.quiet:
            print(
                f"[Timing] sampling={t_after_sampling - t_start:.2f}s, "
                f"postproc+decode+save={t_end - t_after_sampling:.2f}s, "
                f"total={t_end - t_start:.2f}s"
            )
            print(f"[Done] Wrote {args.out}")
        return

    tokens_A = torch.tensor(hist_A, device=tts.device, dtype=torch.long).unsqueeze(0)
    cond_seq = prep["cond_all"][:, :T, :]
    tokens_1xTQ = tts.model.nar_refine(cond_seq, tokens_A)
    tokens_tq = tokens_1xTQ.squeeze(0)

    wav = tts.codec.decode_full(tokens_tq)
    save_audio(args.out, wav, sr=TARGET_SR)

    t_end = time.perf_counter()
    if not args.quiet:
        print(
            f"[Timing] sampling={t_after_sampling - t_start:.2f}s, "
            f"postproc+decode+save={t_end - t_after_sampling:.2f}s, "
            f"total={t_end - t_start:.2f}s"
        )
        print(f"[Done] Wrote {args.out}")


if __name__ == "__main__":
    main()
