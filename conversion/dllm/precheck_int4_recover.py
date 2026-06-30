# Community port — NOT an Apple model.
"""int4 recovery precheck: does CLIPPING (the engine's `symmetric_with_clipping`) or a smaller block
rescue int4 for LLaDA-8B, and is the int4 output actually broken or just a different (valid) diffusion
path? Prints the generated text per config so we can judge coherence, not only token-match.

  coreai-models/.venv/bin/python precheck_int4_recover.py [--gen 64]
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


def fq(w, nbits, block, clip=1.0):
    out_f, in_f = w.shape
    qmax = (1 << (nbits - 1)) - 1
    wb = w.reshape(out_f, in_f // block, block).float()
    scale = (wb.abs().amax(dim=-1, keepdim=True) * clip) / qmax
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(wb / scale), -qmax, qmax)
    return (q * scale).reshape(out_f, in_f).to(w.dtype)


def quantize(model, bits, block, clip):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name != "ff_out":      # body only; head stays fp16
            mod.weight.data = fq(mod.weight.data, bits, block, clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--prompt", type=str,
                    default="Natalia sold clips to 48 friends in April, then half as many in May. "
                            "How many clips did she sell altogether? Reason step by step.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long).unsqueeze(0)
    L = ids.shape[1]
    canvas = torch.full((1, L + args.gen), MASK_ID, dtype=torch.long); canvas[:, :L] = ids
    buf = L + args.gen

    sd = load_weights(args.layers)
    ref = load_backbone(sd, args.layers, buf, torch.float32)
    with torch.inference_mode():
        ref_logits = ref(canvas)
    ref_x, ref_nfe = my_generate(lambda z: ref(z), ids, gen_length=args.gen, block_length=32,
                                 threshold=0.5, mask_id=MASK_ID)
    ref_tokens = ref_x[0, L:]
    print(f"fp32 (nfe={ref_nfe}): {tok.decode(ref_tokens, skip_special_tokens=True)!r}\n")
    del ref

    # block32 with clipping sweep, + block16 absmax
    configs = [
        ("int4 b32 clip1.00", 4, 32, 1.00),
        ("int4 b32 clip0.90", 4, 32, 0.90),
        ("int4 b32 clip0.80", 4, 32, 0.80),
        ("int4 b32 clip0.70", 4, 32, 0.70),
        ("int4 b16 clip1.00", 4, 16, 1.00),
        ("int4 b16 clip0.85", 4, 16, 0.85),
    ]
    for label, bits, block, clip in configs:
        m = load_backbone(sd, args.layers, buf, torch.float32)
        quantize(m, bits, block, clip)
        with torch.inference_mode():
            ql = m(canvas)
        lc = cos(ql[:, L:], ref_logits[:, L:])
        agree = (ql[0, L:].argmax(-1) == ref_logits[0, L:].argmax(-1)).float().mean().item() * 100
        q_x, q_nfe = my_generate(lambda z: m(z), ids, gen_length=args.gen, block_length=32,
                                 threshold=0.5, mask_id=MASK_ID)
        q_tokens = q_x[0, L:]
        match = int((q_tokens == ref_tokens).sum())
        print(f"[{label}] cos={lc:.5f} argmax={agree:.1f}% match={match}/{args.gen} nfe={q_nfe}")
        print(f"   {tok.decode(q_tokens, skip_special_tokens=True)!r}\n")
        del m


if __name__ == "__main__":
    main()
