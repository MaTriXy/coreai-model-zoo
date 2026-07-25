# Community port — NOT an Apple model.
"""Torch cross-check: the static-KV Qwen2 backbone (backbone.py, decode/prefill) must reproduce
the from-scratch full-seq overlay (torch_overlays.Qwen2Backbone) — which is already gated
cos=1.000000 vs the oracle. This closes the KV-cache reformulation before the engine export.

  PYTHONPATH=. <coreai-venv>/bin/python gate_backbone_static.py --src <weights/dots.tts-soar>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from torch_overlays import Qwen2Backbone as FullSeq
from backbone import Qwen2Backbone as StaticKV, Qwen2Cfg, build_kv_state, load_backbone


def cos(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--buf", type=int, default=512)
    a = ap.parse_args()
    DT = torch.float32

    sd = load_file(str(Path(a.src) / "model.safetensors"))
    lm = json.load(open(Path(a.src) / "llm_config.json"))

    # full-seq overlay (the oracle-gated reference)
    cfg_ov = dict(hidden_size=lm["hidden_size"], intermediate_size=lm["intermediate_size"],
                  num_hidden_layers=lm["num_hidden_layers"], num_attention_heads=lm["num_attention_heads"],
                  num_key_value_heads=lm["num_key_value_heads"], rms_norm_eps=lm["rms_norm_eps"],
                  rope_theta=lm["rope_theta"], vocab_size=lm["vocab_size"])
    full = FullSeq(cfg_ov).to(DT).eval()
    full.load_upstream(sd)

    # static-KV backbone
    stat = load_backbone(sd, a.buf, DT)

    torch.manual_seed(0)
    T = a.seq
    embeds = torch.randn(1, T, lm["hidden_size"], dtype=DT)

    with torch.inference_mode():
        h_full, _ = full(embeds)                       # [1,T,H]

        # (A) decode one-by-one from a fresh cache
        kc, vc = build_kv_state(stat.cfg, a.buf, DT)
        h_dec = []
        for i in range(T):
            h_dec.append(stat.decode(embeds[:, i:i + 1], torch.tensor([i], dtype=torch.int32), kc, vc))
        h_dec = torch.cat(h_dec, dim=1)                # [1,T,H]

        # (B) prefill the whole sequence in one call
        kc2, vc2 = build_kv_state(stat.cfg, a.buf, DT)
        h_pre = stat.prefill(embeds, kc2, vc2)         # [1,T,H]

    per_dec = [cos(h_dec[:, i], h_full[:, i]) for i in range(T)]
    per_pre = [cos(h_pre[:, i], h_full[:, i]) for i in range(T)]
    print(f"decode  per-pos min cos = {min(per_dec):.6f}  (last {per_dec[-1]:.6f})")
    print(f"prefill per-pos min cos = {min(per_pre):.6f}  (last {per_pre[-1]:.6f})")
    ok = min(per_dec) >= 0.999 and min(per_pre) >= 0.999
    print(">>>", "STATIC-KV GATE PASS" if ok else "STATIC-KV GATE FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
