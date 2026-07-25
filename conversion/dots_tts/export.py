# Community port — NOT an Apple model.
"""Export the dots.tts Qwen2.5-1.5B backbone to a Core AI .aimodel + engine-gate vs the
static-KV torch backbone (which is itself cos=1.000000 vs the oracle-gated overlay — so it
IS the reference). Mirrors voxcpm/export_v2.py:gate_backbone, scaled to Qwen2 (qkv-bias,
head_dim 128, 28L, GQA 2).

Bundle: inputs_embeds -> hidden, static KV (data-driven write slot, whole-buffer read,
runtime causal mask). decode (q=1, runtime pos) [+ optional q=N prefill].

  PYTHONPATH=. <coreai-venv>/bin/python export.py --which backbone --mode int8 \
      --src <weights/dots.tts-soar> --cache-len 512 [--prefill-len 32]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402
from backbone import Qwen2Cfg, build_kv_state, load_backbone  # noqa: E402

_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
ART = Path(__file__).resolve().parent / "artifacts"


def quant_cfg(dtype: str) -> dict:
    """Weight-only linear per-block-32 symmetric-with-clipping (GPU scale-multiply dequant,
    no LUT). Excludes SDPA/RMSNorm/RoPE primitives + Embedding/Conv (none here anyway)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            # hand-rolled norm: its rank-1 `.weight` breaks per-block axis-1 — keep it fp
            "backbone.RMSNorm": None,
        },
    }


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    print(f"  saving {aim.name} ...", flush=True)
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


class DecodeWrap(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb

    def forward(self, inputs_embeds, pos, k_cache, v_cache):
        return self.bb.decode(inputs_embeds, pos, k_cache, v_cache)


class PrefillWrap(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb

    def forward(self, inputs_embeds, k_cache, v_cache):
        return self.bb.prefill(inputs_embeds, k_cache, v_cache)


async def gate_backbone(DT, sd, cache_len, prefill_len, mode):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    int_q = mode in ("int8", "int4")
    bb = load_backbone(sd, cache_len, DT)
    H = bb.cfg.hidden_size
    nl, NKV, HD = bb.cfg.num_hidden_layers, bb.cfg.num_key_value_heads, bb.cfg.head_dim
    kc, vc = build_kv_state(bb.cfg, cache_len, DT)

    # ---- decode bundle ----
    dec_ref = {"inputs_embeds": torch.zeros(1, 1, H, dtype=DT), "pos": torch.tensor([0], dtype=torch.int32),
               "k_cache": kc.clone(), "v_cache": vc.clone()}
    dec_model = DecodeWrap(bb).eval()
    if int_q:
        print(f"  [{mode}] weight-only quantize (decode) ...", flush=True)
        dec_model = quantize_pytorch_model(
            dec_model, (dec_ref["inputs_embeds"], dec_ref["pos"], dec_ref["k_cache"], dec_ref["v_cache"]),
            {"inputs_embeds": None, "pos": None, "k_cache": None, "v_cache": None}, quant_cfg(mode))
    prog = export_to_coreai(dec_model, dec_ref, dynamic_shapes=None,
                            input_names=("inputs_embeds", "pos"), output_names=("hidden",),
                            state_names=("keyCache", "valueCache"))
    ddir = ART / f"dots_backbone_{mode}_decode_cl{cache_len}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir.name} ({_du(aim)})", flush=True)

    # gate: thread K decode steps, torch static-KV vs engine-state
    K = 6
    torch.manual_seed(0)
    embs = [torch.randn(1, 1, H, dtype=DT) for _ in range(K)]
    tk, tv = build_kv_state(bb.cfg, cache_len, DT)
    t_hid = []
    with torch.inference_mode():
        for i, e in enumerate(embs):
            t_hid.append(bb.decode(e, torch.tensor([i], dtype=torch.int32), tk, tv).reshape(-1).float())
    dfn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    state = {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16)),
             "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16))}
    cs = []
    for i, e in enumerate(embs):
        r = await dfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(e.numpy().astype(np.float16))),
                              "pos": rt.NDArray(np.ascontiguousarray(np.array([i], np.int32)))}, state=state)
        cs.append(cos(r["hidden"].numpy(), t_hid[i]))
    print(f"  decode engine-vs-torch min cos={min(cs):.6f}  {['%.4f' % c for c in cs]}")

    # ---- prefill bundle (q=prefill_len) ----
    if prefill_len:
        pref_ref = {"inputs_embeds": torch.zeros(1, prefill_len, H, dtype=DT),
                    "k_cache": kc.clone(), "v_cache": vc.clone()}
        pre_model = PrefillWrap(bb).eval()
        if int_q:
            print(f"  [{mode}] weight-only quantize (prefill q={prefill_len}) ...", flush=True)
            pre_model = quantize_pytorch_model(
                pre_model, (pref_ref["inputs_embeds"], pref_ref["k_cache"], pref_ref["v_cache"]),
                {"inputs_embeds": None, "k_cache": None, "v_cache": None}, quant_cfg(mode))
        progp = export_to_coreai(pre_model, pref_ref, dynamic_shapes=None,
                                 input_names=("inputs_embeds",), output_names=("hidden",),
                                 state_names=("keyCache", "valueCache"))
        pdir = ART / f"dots_backbone_{mode}_prefill_t{prefill_len}"
        aimp = _save(progp, pdir)
        print(f"  -> {pdir.name} ({_du(aimp)})", flush=True)
        torch.manual_seed(1)
        pe = torch.randn(1, prefill_len, H, dtype=DT)
        pk, pv = build_kv_state(bb.cfg, cache_len, DT)
        with torch.inference_mode():
            t_p = bb.prefill(pe, pk, pv).reshape(-1).float()
        pfn = (await rt.AIModel.load(str(aimp), gpu)).load_function("main")
        pstate = {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16)),
                  "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16))}
        rp = await pfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(pe.numpy().astype(np.float16)))},
                       state=pstate)
        cs.append(cos(rp["hidden"].numpy(), t_p))
        print(f"  prefill q={prefill_len} engine-vs-torch cos={cs[-1]:.6f}")
    return min(cs)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="backbone", choices=["backbone"])
    ap.add_argument("--mode", default="int8", choices=["fp16", "int8", "int4"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--prefill-len", type=int, default=0)
    a = ap.parse_args()
    DT = torch.float16
    ART.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(Path(a.src) / "model.safetensors"))

    lo = await gate_backbone(DT, sd, a.cache_len, a.prefill_len, a.mode)
    print(f"\n>>> {a.which}/{a.mode} export+engine: cos={lo:.6f} -> {'GATE PASS' if lo >= 0.99 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
