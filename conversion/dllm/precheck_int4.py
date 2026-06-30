# Community port — NOT an Apple model.
"""int4 quantization-quality precheck for the LLaDA-8B dLLM port (go/no-go before full export).

LLaDA/d3LLM ships bf16 (NOT QAT-int4), so naive PTQ-int4 risks the "int4 cliff" (cf BitCPM ternary /
LFM int4 lessons). This does a cheap fake-quant ROUND-TRIP (quantize -> dequantize the Linear weights
in place, run the fp32 overlay) which faithfully reflects weight-only quant error on the logits AND on
the actual denoising output, without the export backend. Mirrors the zoo int4 ship-shape:

  body  = int4 weight-only, per-block-32 (axis=1, in-features), symmetric    [int4lin]
  head  = lm_head (ff_out, 126464-vocab, fat-tailed) kept fp16, or int8 absmax  [int4hu]
  embed = wte kept fp16 (gather, not a matmul)

Reports logits cos + masked-argmax agreement vs fp32, and re-runs the decode loop to check the
generated tokens still match fp32. int8 per-block-32 is the safe baseline for comparison.

  coreai-models/.venv/bin/python precheck_int4.py [--layers 32 --gen 64]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "d3LLM_LLaDA")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref"))

from llada import load_backbone                                    # noqa: E402
from generate_llada import generate as my_generate                # noqa: E402
from gate_llada_torch import load_weights                         # noqa: E402

MASK_ID = 126336


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def fake_quant_blockwise(w: torch.Tensor, nbits: int, block: int = 32) -> torch.Tensor:
    """Weight-only symmetric per-block-`block` fake-quant along axis=1 (in-features). absmax scale.

    absmax (no clipping) is a conservative lower bound on quality vs the engine's
    `symmetric_with_clipping` (clipping reduces outlier impact) — if absmax holds, ship-shape holds.
    """
    out_f, in_f = w.shape
    assert in_f % block == 0, f"{in_f} not divisible by {block}"
    qmax = (1 << (nbits - 1)) - 1                       # int4 -> 7, int8 -> 127
    wb = w.reshape(out_f, in_f // block, block).float()
    scale = wb.abs().amax(dim=-1, keepdim=True) / qmax  # [out, in/block, 1]
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(wb / scale), -qmax, qmax)
    return (q * scale).reshape(out_f, in_f).to(w.dtype)


def quantize_overlay(model: nn.Module, body_bits: int, head_bits, block: int = 32):
    """Round-trip the body Linear weights to `body_bits`; lm_head (ff_out) to `head_bits`
    (None = keep fp). wte (Embedding) untouched. Mutates in place."""
    n_body = n_head = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        is_head = name == "ff_out"                       # top-level lm_head
        if is_head:
            if head_bits is None:
                continue
            mod.weight.data = fake_quant_blockwise(mod.weight.data, head_bits, block)
            n_head += 1
        else:
            mod.weight.data = fake_quant_blockwise(mod.weight.data, body_bits, block)
            n_body += 1
    return n_body, n_head


def build_canvas(tok, prompt: str, gen: int):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long).unsqueeze(0)
    L = ids.shape[1]
    x = torch.full((1, L + gen), MASK_ID, dtype=torch.long)
    x[:, :L] = ids
    return ids, x, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--prompt", type=str,
                    default="Natalia sold clips to 48 friends in April, then half as many in May. "
                            "How many clips did she sell altogether? Reason step by step.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids, canvas, L = build_canvas(tok, args.prompt, args.gen)
    buf = L + args.gen
    print(f"[precheck] layers={args.layers} prompt_len={L} gen={args.gen} block={args.block}")

    sd = load_weights(args.layers)

    # fp32 reference (logits on the canvas + decode tokens)
    ref = load_backbone(sd, args.layers, buf, torch.float32)
    with torch.inference_mode():
        ref_logits = ref(canvas)
    ref_x, ref_nfe = my_generate(lambda z: ref(z), ids, gen_length=args.gen,
                                 block_length=args.block, threshold=0.5, mask_id=MASK_ID)
    ref_tokens = ref_x[0, L:]
    ref_text = tok.decode(ref_tokens, skip_special_tokens=True)
    print(f"  fp32 decode: nfe={ref_nfe}  {ref_text!r}")
    del ref

    configs = [
        ("int8  body / fp16 head", 8, None),
        ("int4  body / fp16 head", 4, None),   # int4lin
        ("int4  body / int8 head", 4, 8),      # int4hu  (ship target ~4-5GB)
    ]
    print(f"\n  {'config':26s} {'logits cos':>11s} {'mask-argmax':>12s} {'decode':>22s}")
    for label, bb, hb in configs:
        m = load_backbone(sd, args.layers, buf, torch.float32)
        nb, nh = quantize_overlay(m, bb, hb, args.block)
        with torch.inference_mode():
            q_logits = m(canvas)
        lc = cos(q_logits[:, L:], ref_logits[:, L:])
        my_pred = q_logits[0, L:].argmax(-1)
        rf_pred = ref_logits[0, L:].argmax(-1)
        agree = (my_pred == rf_pred).float().mean().item() * 100
        q_x, q_nfe = my_generate(lambda z: m(z), ids, gen_length=args.gen,
                                 block_length=args.block, threshold=0.5, mask_id=MASK_ID)
        q_tokens = q_x[0, L:]
        tok_match = int((q_tokens == ref_tokens).sum())
        same = bool((q_tokens == ref_tokens).all())
        dec = f"{tok_match}/{args.gen} nfe={q_nfe}{' ✓' if same else ''}"
        print(f"  {label:26s} {lc:11.6f} {agree:11.1f}% {dec:>22s}")
        del m

    print(f"\n  fp32 ref text: {ref_text!r}")
    print("  (judge by decode token-match + coherent text; logits cos is the fast proxy.)")


if __name__ == "__main__":
    main()
