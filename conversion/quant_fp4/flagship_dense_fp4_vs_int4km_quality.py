#!/usr/bin/env python3
"""Flagship (Qwen3.6-35B-A3B) dense-path QUALITY gate: fp4 vs int4km vs int8km on the REAL weights.

The dense-path lever quantizes lm_head + full-attn q/o. The two flagship bundles (fp4, int4km) share
identical routed experts (gather_qmm km4) and int8 non-expert linears, so their quality difference comes
ONLY from the dense path -- above all **lm_head**, which directly produces the output tokens (the
"~12/41 token-flip" the int4 recipe article reported = an lm_head effect, the Nanbeige lesson).

int4km here is the SAME scheme the flagship int4km bundle uses (``palettize_grouped`` n_bits=4, 16-entry
k-means codebook per 32-output-row group) -- NOT weak int4-RTN. fp4 is the SAME scheme the fp4 kernel
uses (``quantize_fp4_e2m1``, verified bit-identical to torchao/coreai-opt fp4). So this is the faithful
fp4-vs-int4km comparison on the exact flagship weights.

Metrics (no model forward needed -- weights loaded directly from the safetensors shards):
  1. Weight reconstruction relative error per scheme.
  2. lm_head top-1 token-FLIP rate vs fp16, on realistic hidden states (embedding rows passed through the
     model's final RMSNorm -- in-distribution vectors, not random gaussian). Also top-1 logit agreement.

Run:  cd coreai-models && .venv/bin/python \
        ../coreai-models-community/conversion/quant_fp4/flagship_dense_fp4_vs_int4km_quality.py [--samples 512]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch
from safetensors import safe_open

from coreai_models.models.macos.gemma4_metal_mlp import palettize_grouped, _unpack_nib
from coreai_models.models.macos.gemma4_metal_mlp_fp4 import quantize_fp4_e2m1, FP4_LUT

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*"))[0]


def load_tensor(key: str) -> torch.Tensor:
    wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    shard = wm[key]
    with safe_open(os.path.join(SNAP, shard), framework="pt") as f:
        return f.get_tensor(key)


def recon_fp4(W):
    qp, sc = quantize_fp4_e2m1(W)
    G = (qp.shape[1] * 8) // sc.shape[1]
    vals = FP4_LUT[_unpack_nib(qp)]
    return vals * sc.float().repeat_interleave(G, dim=1)


def recon_fp4_f16scale(W, block=32):
    """fp4 with a full-fp16 per-block absmax scale (NOT e8m0) — the most accurate fp4 this hand-rolled
    matvec can use (it reads SC as fp16; e8m0 is only required by OS27 TensorOps, unused here). Included
    to show that even the best-scaled fp4 does not beat int4km on token flips."""
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, f4_unpacked_to_f32
    N, K = W.shape
    Wb = W.float().reshape(N, K // block, block)
    amax = Wb.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    sc = (amax / 6.0).half().float()
    dq = f4_unpacked_to_f32(f32_to_f4_unpacked((Wb / sc).clamp(-6, 6)))
    return (dq * sc).reshape(N, K)


def recon_km(W, n_bits):
    idx, cb = palettize_grouped(W, n_bits=n_bits, iters=10)
    grp = torch.arange(W.shape[0]) // 32
    return cb[grp.unsqueeze(1), idx.long()].float()


def rel_err(Wq, W):
    return (Wq - W.float()).norm().item() / (W.float().norm().item() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=512)
    args = ap.parse_args()
    torch.manual_seed(0)

    cfg = json.load(open(os.path.join(SNAP, "config.json")))
    tc = cfg.get("text_config", cfg)
    eps = tc.get("rms_norm_eps", 1e-6)

    print("== Flagship Qwen3.6-35B dense-path quality: fp4 vs int4km vs int8km (real weights) ==\n")

    # ---- reconstruction error on the dense-path weights ----
    targets = {
        "lm_head": "lm_head.weight",
        "attn.q_proj(L11)": "model.language_model.layers.11.self_attn.q_proj.weight",
        "attn.o_proj(L11)": "model.language_model.layers.11.self_attn.o_proj.weight",
    }
    print(f"{'weight':18} {'shape':>16} {'int8km':>9} {'int4km':>9} {'fp4_e8m0':>9} {'fp4_f16s':>9}")
    lm_head = None
    for label, key in targets.items():
        W = load_tensor(key).float()
        if label == "lm_head":
            lm_head = W
        e8 = rel_err(recon_km(W, 8), W)
        e4 = rel_err(recon_km(W, 4), W)
        ef = rel_err(recon_fp4(W), W)
        eff = rel_err(recon_fp4_f16scale(W), W)
        print(f"{label:18} {str(tuple(W.shape)):>16} {e8:>9.4f} {e4:>9.4f} {ef:>9.4f} {eff:>9.4f}")

    # ---- lm_head top-1 token-flip vs fp16 on realistic hidden states ----
    print("\n-- lm_head top-1 token-flip vs fp16 (hidden = embed rows through final RMSNorm) --")
    embed = load_tensor("model.language_model.embed_tokens.weight").float()
    norm_w = load_tensor("model.language_model.norm.weight").float()
    S = min(args.samples, embed.shape[0])
    sel = torch.randperm(embed.shape[0])[:S]
    h = embed[sel]
    h = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w  # RMSNorm (no +1)

    def logits(W):
        return h @ W.t()

    ref = logits(lm_head).argmax(-1)
    print(f"   samples: {S}")
    print(f"   {'scheme':8} {'flip vs fp16':>13} {'top1-agree':>11}")
    schemes = {"int8km": recon_km(lm_head, 8), "int4km": recon_km(lm_head, 4),
               "fp4_e8m0": recon_fp4(lm_head), "fp4_f16scale": recon_fp4_f16scale(lm_head)}
    for nm, Wq in schemes.items():
        pred = logits(Wq).argmax(-1)
        flips = int((pred != ref).sum())
        print(f"   {nm:8} {flips:>6}/{S:<6} {(1-flips/S)*100:>9.2f}%")
    print("\nGate: fp4 flip-rate should track int8km (quality-safe) and beat int4km — that is the")
    print("      quality reason to choose fp4 over int4km at the SAME ¼-byte decode bandwidth.")


if __name__ == "__main__":
    main()
