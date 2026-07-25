#!/usr/bin/env python
"""P2: dump the RawLoop weights (already in KERNEL layout) to a flat pack for Swift.

Output: pack/gemma4_pack.bin (64B-aligned tensors) + pack/gemma4_pack.json
(offsets/dtypes/shapes + the per-layer routing/meta table). The Swift runner
mmaps the blob into ONE bytesNoCopy MTLBuffer and binds tensors by offset.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import p1_chain as p1

p1.DEV = "cpu"  # build all kernel-layout tensors on CPU (RawLoop reads module global)

HERE = Path(__file__).resolve().parent
OUT = HERE / "pack"
ALIGN = 64

DTYPES = {
    torch.float16: "f16", torch.float32: "f32",
    torch.int32: "i32", torch.uint8: "u8", torch.int8: "i8",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    from p3_mtp import RawLoopMTP
    raw = RawLoopMTP(p1.Extract(p1.EXTRACT))
    blob = bytearray()
    tensors: dict[str, dict] = {}

    def add(name: str, t: torch.Tensor) -> None:
        t = t.contiguous().cpu()
        pad = (-len(blob)) % ALIGN
        blob.extend(b"\0" * pad)
        data = t.numpy().tobytes()
        tensors[name] = {"offset": len(blob), "nbytes": len(data),
                         "dtype": DTYPES[t.dtype], "shape": list(t.shape)}
        blob.extend(data)

    add("embed.packed", raw.emb_packed)
    add("embed.scale", raw.emb_scale)
    add("ple.packed", raw.ple_packed)
    add("ple.scale", raw.ple_scale)
    add("model_proj.w8", raw.model_proj.w8)
    add("model_proj.sc", raw.model_proj.sc)
    add("proj_norm", raw.proj_norm)
    add("final_norm", raw.final_norm)
    add("lm_head.qp", raw.lm_head.qp)
    add("lm_head.sc", raw.lm_head.sc)
    add("invf.sliding", raw.invf_sliding)
    add("invf.full", raw.invf_full)

    layers_meta = []
    for li, lay in enumerate(raw.layers):
        P = f"L{li:02d}."
        add(P + "wq.qp", lay.wq.qp); add(P + "wq.sc", lay.wq.sc); add(P + "wq.bi", lay.wq.bi)
        add(P + "wo.qp", lay.wo.qp); add(P + "wo.sc", lay.wo.sc); add(P + "wo.bi", lay.wo.bi)
        if lay.write:
            add(P + "wk.qp", lay.wk.qp); add(P + "wk.sc", lay.wk.sc); add(P + "wk.bi", lay.wk.bi)
            add(P + "wv.qp", lay.wv.qp); add(P + "wv.sc", lay.wv.sc); add(P + "wv.bi", lay.wv.bi)
            add(P + "key_norm", lay.key_norm)
        for nm, w in (("gate", lay.wgate), ("up", lay.wup), ("down", lay.wdown)):
            add(P + nm + ".qp", w.qp)
            add(P + nm + ".sc", w.sc)
            if not lay.int2:
                add(P + nm + ".bi", w.bi)
        add(P + "ple_gate.w8", lay.ple_gate.w8); add(P + "ple_gate.sc", lay.ple_gate.sc)
        add(P + "ple_proj.w8", lay.ple_proj.w8); add(P + "ple_proj.sc", lay.ple_proj.sc)
        add(P + "pre_attn", lay.pre_attn); add(P + "post_attn", lay.post_attn)
        add(P + "pre_ffw", lay.pre_ffw); add(P + "post_ffw", lay.post_ffw)
        add(P + "post_ple", lay.post_ple); add(P + "query_norm", lay.query_norm)
        layers_meta.append({
            "full": lay.full, "write": lay.write, "cache": lay.cache,
            "int2": lay.int2, "hd": lay.hd, "layer_scalar": lay.layer_scalar,
            "ffn_n": lay.wgate.N, "ffn_k": lay.wdown.K,
        })

    # ---- MTP drafter (kernel layouts already built by RawLoopMTP) ----
    add("dft.pre_proj.w8", raw.d_pre_proj.w8)
    add("dft.pre_proj.sc", raw.d_pre_proj.sc)
    add("dft.post_proj.w8", raw.d_post_proj_w8)
    add("dft.post_proj.sc", raw.d_post_proj_sc)
    add("dft.head.qp", raw.d_head.qp)
    add("dft.head.sc", raw.d_head.sc)
    add("dft.head.bi", raw.d_head.bi)
    add("dft.final_norm", raw.d_final_norm)
    add("dft.invf128", raw.d_invf[256])
    add("dft.invf256", raw.d_invf[512])
    dft_layers = []
    for i, dl in enumerate(raw.d_layers):
        P = f"dft.L{i}."
        add(P + "wq.w8", dl.wq.w8); add(P + "wq.sc", dl.wq.sc)
        add(P + "wo.w8", dl.wo.w8); add(P + "wo.sc", dl.wo.sc)
        for nm, w in (("gate", dl.wgate), ("up", dl.wup), ("down", dl.wdown)):
            add(P + nm + ".qp", w.qp); add(P + nm + ".sc", w.sc); add(P + nm + ".bi", w.bi)
        add(P + "pre_attn", dl.pre_attn); add(P + "post_attn", dl.post_attn)
        add(P + "pre_ffw", dl.pre_ffw); add(P + "post_ffw", dl.post_ffw)
        add(P + "q_norm", dl.q_norm)
        dft_layers.append({
            "hd": dl.hd, "cache": dl.cache, "skip": raw.d_skip[i],
            "aq_q": dl.aq_q, "aq_attn_vec": dl.aq_attn_vec,
            "aq_gating": dl.aq_gating, "aq_down": dl.aq_down,
        })

    manifest = {
        "tensors": tensors,
        "meta": {
            "vocab": raw.V, "hidden": p1.HID, "layers": layers_meta,
            "window": p1.WINDOW, "max_ctx": p1.MAX_CTX,
            "pli_scale": p1.PLI_SCALE, "n_heads": p1.H,
            "drafter": {
                "layers": dft_layers,
                "aq_pre_proj": raw.aq_pre_proj, "aq_post_proj": raw.aq_post_proj,
            },
        },
    }
    (OUT / "gemma4_pack.bin").write_bytes(bytes(blob))
    (OUT / "gemma4_pack.json").write_text(json.dumps(manifest))
    print(f"pack: {len(blob) / 1e9:.2f} GB, {len(tensors)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
