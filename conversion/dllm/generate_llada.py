# Community port — NOT an Apple model.
"""LLaDA / d3LLM masked-diffusion DECODE loop (host side), clean reimplementation.

The model is a mask predictor: given a canvas `x` (prompt followed by [MASK] tokens), one full
bidirectional forward yields logits at every position; we UNMASK (commit) the most-confident masked
positions and repeat. This file ports the official `generate` loop (the canonical no-cache path in
`_ref/d3LLM/.../d3llm_llada_generate_util.py`, used in its `main()`):

* semi-autoregressive BLOCKS of `block_length` — only positions up to the current block's end are
  eligible to decode each step (future blocks stay masked).
* per-step unmask set = masked positions ranked by LOW ENTROPY (= high confidence); always keep the
  single lowest-entropy one (guaranteed progress) plus any others with entropy <= `threshold`.
* temperature 0 => argmax (deterministic); entropy/softmax in float64 to match the reference exactly.

`forward` is any callable `x[1,T] (long) -> logits[1,T,vocab]` — the overlay (`llada.LLaDABackbone`)
in torch, or later the exported engine. NO KV cache here: every step is a full forward over the whole
canvas. The d3LLM delayed-KV-cache speedup (`generate_multi_block_kv_cache`) is a device-side
optimization layered on top of this same decoding logic and is ported separately.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MASK_ID = 126336


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. temperature 0 => identity (argmax). float64 per the LLaDA paper."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def transfer_by_entropy(logits, mask_index, x, temperature, entropy_threshold):
    """Pick which masked positions to unmask this step (entropy-thresholded low-confidence remasking).

    Returns (x0, transfer_index): x0 = predicted tokens (argmax over masked, else current x);
    transfer_index = bool[1,T] of positions to commit. Always commits the lowest-entropy masked
    position; additionally commits any whose entropy <= threshold.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)                       # [1,T]

    p = F.softmax(logits.to(torch.float64), dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)            # [1,T]

    x0 = torch.where(mask_index, x0, x)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    num = mask_index.sum(dim=1, keepdim=True)                          # all masked positions are candidates
    for j in range(entropy_for_selection.shape[0]):
        k = int(num[j])
        if k == 0:
            continue
        _, select_index = torch.topk(entropy_for_selection[j], k=k, largest=False)  # lowest entropy first
        transfer_index[j, select_index] = True
        for r in range(1, k):                                          # keep r=0 (lowest), threshold the rest
            if entropy[j, select_index[r]] > entropy_threshold:
                transfer_index[j, select_index[r]] = False
    return x0, transfer_index


@torch.no_grad()
def generate(forward, prompt, gen_length=256, block_length=32, temperature=0.0,
             threshold=0.5, mask_id=MASK_ID, eos_token_id=None):
    """Semi-autoregressive block diffusion decode. Mirrors the official no-cache `generate`.

    forward: callable x[1,T] long -> logits[1,T,vocab].  prompt: [1,L] long.
    Returns (x[1,L+gen_length], nfe).
    """
    device = prompt.device
    L = prompt.shape[1]
    x = torch.full((1, L + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt.clone()

    # ceil division; the final block may be a partial < block_length (clamped, like the official
    # generate_multi_block). When gen_length % block_length == 0 this is the canonical `generate`.
    num_blocks = (gen_length + block_length - 1) // block_length
    nfe = 0
    for nb in range(num_blocks):
        blk_lo = L + nb * block_length
        blk_hi = min(L + (nb + 1) * block_length, L + gen_length)
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = forward(x)
            mask_index[:, blk_hi:] = False                # do not decode future blocks
            x0, transfer_index = transfer_by_entropy(logits, mask_index, x, temperature, threshold)
            x[transfer_index] = x0[transfer_index]
            if (x[:, blk_lo:blk_hi] == mask_id).sum() == 0:
                break
            if nfe > 10000:
                return x, nfe
        if eos_token_id is not None and (x[:, L:blk_hi] == eos_token_id).any():
            # everything after the first EOS is end-of-text; stop early
            break
    return x, nfe
