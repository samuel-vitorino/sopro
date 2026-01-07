from __future__ import annotations

from typing import List

import torch


def center_crop_tokens(ref_tq: torch.Tensor, win_frames: int) -> torch.Tensor:
    T = int(ref_tq.size(0))
    if T <= win_frames:
        return ref_tq
    s = (T - win_frames) // 2
    return ref_tq[s : s + win_frames]


def repeated_tail(hist: List[int], max_n: int = 16) -> bool:
    L = len(hist)
    for n in range(3, min(max_n, L // 2) + 1):
        if hist[-n:] == hist[-2 * n : -n]:
            return True
    return False


def sample_token(
    logits_1x1v: torch.Tensor,
    history: List[int],
    top_p: float = 0.9,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    eps: float = 1e-12,
) -> int:
    x = logits_1x1v

    x = torch.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)

    if temperature and temperature != 1.0:
        x = x / float(temperature)

    if repetition_penalty != 1.0 and len(history) > 0:
        context = history[-50:]
        ids = torch.tensor(list(set(context)), device=x.device, dtype=torch.long)
        if ids.numel() > 0:
            vals = x[0, 0, ids]
            neg = vals < 0
            vals = torch.where(
                neg, vals * repetition_penalty, vals / repetition_penalty
            )
            x = x.clone()
            x[0, 0, ids] = vals

    probs = torch.softmax(x, dim=-1).view(1, -1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    V = int(probs.size(-1))
    if top_k and top_k > 0:
        k = min(int(top_k), V)
        val, idx = torch.topk(probs, k, dim=-1)
        newp = torch.zeros_like(probs)
        newp.scatter_(1, idx, val)
        probs = newp

        s = probs.sum(dim=-1, keepdim=True)
        if float(s.item()) <= eps:
            return int(torch.argmax(x[0, 0]).item())
        probs = probs / s

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        remove = cum > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove, 0.0)

        s = sorted_probs.sum(dim=-1, keepdim=True)
        if float(s.item()) <= eps:
            return int(torch.argmax(x[0, 0]).item())
        sorted_probs = sorted_probs / s

        j = torch.multinomial(sorted_probs, 1).item()
        token = int(sorted_idx[0, j].item())

        return token

    s = probs.sum(dim=-1, keepdim=True)
    if float(s.item()) <= eps:
        return int(torch.argmax(x[0, 0]).item())
    probs = probs / s

    return int(torch.multinomial(probs, 1).item())


def rf_ar(ar_kernel: int, dilations: Tuple[int, ...]) -> int:
    return 1 + (ar_kernel - 1) * int(sum(dilations))


def rf_nar(n_layers_nar: int, kernel_size: int = 7, dilation: int = 1) -> int:
    return 1 + (kernel_size - 1) * int(n_layers_nar) * int(dilation)
