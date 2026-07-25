# Community port — NOT an Apple model.
"""Two-stage engine composition for ONE audio patch: DiT solver bundle -> denormalize ->
patch_encoder bundle, gated vs the oracle. Proves the exported Core AI bundles CHAIN on the
engine (not just individually), driven by the host glue — the core of the per-patch loop.

  patch-0 noise -> [engine DiT solver, soar 10-step CFG] -> latent patch (normalized)
       -> denormalize (latent_stats) -> [engine patch_encoder decode_patch, empty cache] -> LLM embed
  gate: engine embed vs oracle patch_encoder.out_embedding.

  PYTHONPATH=. <coreai-venv>/bin/python e2e_patch.py --src <weights/dots.tts-soar>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_overlays import DiTOverlay  # noqa: E402
from host_solver import _linear, DiTEngine, solve_soar, cos  # noqa: E402
from patch_encoder import build_kv_state, load_patch_encoder  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"


async def main_run(src, DT):
    torch.set_grad_enabled(False)
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    src = Path(src)
    sd = load_file(str(src / "model.safetensors"))
    cfg_json = json.loads((src / "config.json").read_text())
    z = np.load(ART / "oracle_ref.npz")

    # --- stage 1: engine DiT solver -> patch (soar) ---
    coord = _linear(sd, "coordinate_proj", DT)
    ov = DiTOverlay(cfg_json["DiT"], mode="flow_matching").to(DT).eval()
    ov.load_upstream({k: v for k, v in sd.items() if k.startswith("velocity_field_predictor.")})
    kx = torch.from_numpy(z["dit.kw_x"]).to(DT)
    cond, uncond = kx[0:1, :1], kx[1:2, :1]
    attn_mask = torch.from_numpy(z["dit.kw_attn_mask"]).to(DT)
    pos_ids = torch.from_numpy(z["dit.kw_pos_ids"]).to(torch.float32)
    g_cond = torch.from_numpy(z["dit.kw_g_cond"]).to(DT)[0:1]
    noise = torch.from_numpy(z["randn0"]).to(DT)
    names = ("x", "timesteps", "attn_mask", "pos_ids", "g_cond")
    aim_dit = ART / "dots_dit_soar_fp16_s5" / "dots_dit_soar_fp16_s5.aimodel"
    dfn = (await rt.AIModel.load(str(aim_dit), gpu)).load_function("main")
    patch = await solve_soar(DiTEngine(dfn, names), coord, cond, uncond, g_cond,
                             attn_mask, pos_ids, noise, DT, engine=True)  # [1,4,128] normalized

    # denormalize
    st = torch.load(str(src / "latent_stats.pt"), weights_only=False)
    mean = torch.as_tensor(st["mean"]).to(DT); std = torch.sqrt(torch.as_tensor(st["var"]).to(DT))
    latent_patch = (patch.to(DT) * std + mean)  # [1,4,128]
    print(f"  stage1 solver patch -> denorm cos vs oracle in_latent_patch = "
          f"{cos(latent_patch.numpy(), z['patch_encoder.in_latent_patch']):.6f}")

    # --- stage 2: engine patch_encoder decode_patch (patch 0: empty cache) ---
    buf = z["patch_encoder.in_layer_caches_0_0"].shape[2]
    _, cfg = load_patch_encoder(sd, cfg_json, buf, DT)
    nl, nh, hd = cfg.n_layers, cfg.n_heads, cfg.head_dim
    conv_tail = torch.from_numpy(z["patch_encoder.in_conv_tail"]).to(DT)  # patch-0 init tail
    pos0 = int(torch.from_numpy(z["patch_encoder.in_positions"])[0].item())
    aim_pe = ART / f"dots_patchenc_fp16_buf{buf}" / f"dots_patchenc_fp16_buf{buf}.aimodel"
    pfn = (await rt.AIModel.load(str(aim_pe), gpu)).load_function("main")
    state = {"keyCache": rt.NDArray(np.zeros((nl, 1, nh, buf, hd), np.float16)),
             "valueCache": rt.NDArray(np.zeros((nl, 1, nh, buf, hd), np.float16))}
    r = await pfn(inputs={
        "latent_patch": rt.NDArray(np.ascontiguousarray(latent_patch.numpy())),
        "conv_tail": rt.NDArray(np.ascontiguousarray(conv_tail.numpy())),
        "pos": rt.NDArray(np.ascontiguousarray(np.array([pos0], np.int32))),
    }, state=state)
    c = cos(r["embedding"].numpy(), z["patch_encoder.out_embedding"])
    print(f"  stage2 patch_encoder embed vs oracle out_embedding = {c:.6f}")
    print(f"\n>>> 2-stage engine chain (DiT solver -> patch_encoder): cos={c:.6f} -> "
          f"{'PASS' if c >= 0.99 else 'FAIL'}")
    sys.exit(0 if c >= 0.99 else 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    await main_run(a.src, torch.float16)


if __name__ == "__main__":
    asyncio.run(main())
