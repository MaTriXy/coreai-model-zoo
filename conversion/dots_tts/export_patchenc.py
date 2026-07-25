# Community port — NOT an Apple model.
"""Export the dots.tts patch_encoder (decode_patch) to a Core AI .aimodel + engine-gate vs the
upstream oracle fixture. Static-KV graph (fp16):

  latent_patch[1,4,128] + conv_tail[1,128,1] + pos(int32[1])  [state: keyCache/valueCache
     [24,1,16,BUF,64]]  ->  embedding[1,1,1536] + new_conv_tail[1,128,1]

Plain causal transformer (no qk_norm, no RoPE — see patch_encoder.py) + a causal Conv1d
downsample. Kept fp16 (continuous-feedback path).

  PYTHONPATH=. <coreai-venv>/bin/python export_patchenc.py --src <weights/dots.tts-soar> [--buf 1000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
from patch_encoder import build_kv_state, load_patch_encoder  # noqa: E402

_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
ART = Path(__file__).resolve().parent / "artifacts"


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


class Wrap(nn.Module):
    """Single-output wrapper (embedding only) so the bundle fits kit `StatefulGraphModel`
    (exactly 2 states + 1 output). new_conv_tail = the last latent frame of the input, which the
    host derives itself (see `_downsample_step`: new_conv_tail = raw[..., -1:])."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, latent_patch, conv_tail, pos, k_cache, v_cache):
        emb, _ = self.m(latent_patch, conv_tail, pos, k_cache, v_cache)
        return emb


async def run(src, buf, DT):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    src = Path(src)
    sd = load_file(str(src / "model.safetensors"))
    cfg_json = json.loads((src / "config.json").read_text())
    m, cfg = load_patch_encoder(sd, cfg_json, buf, DT)
    nl, nh, hd = cfg.n_layers, cfg.n_heads, cfg.head_dim

    z = np.load(ART / "oracle_ref.npz")
    latent = torch.from_numpy(z["patch_encoder.in_latent_patch"]).to(DT)
    conv_tail = torch.from_numpy(z["patch_encoder.in_conv_tail"]).to(DT)
    pos0 = int(torch.from_numpy(z["patch_encoder.in_positions"])[0].item())

    kc, vc = build_kv_state(cfg, buf, DT)
    ref = {"latent_patch": latent, "conv_tail": conv_tail,
           "pos": torch.tensor([pos0], dtype=torch.int32), "k_cache": kc, "v_cache": vc}

    prog = export_to_coreai(Wrap(m).eval(), ref, dynamic_shapes=None,
                            input_names=("latent_patch", "conv_tail", "pos"),
                            output_names=("embedding",),
                            state_names=("keyCache", "valueCache"))
    ddir = ART / f"dots_patchenc_fp16_buf{buf}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir.name} ({_du(aim)})", flush=True)

    # engine gate: seed state from the oracle PRE-write caches, run at pos0, compare emb+tail
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    kstack = np.zeros((nl, 1, nh, buf, hd), np.float16)
    vstack = np.zeros((nl, 1, nh, buf, hd), np.float16)
    for i in range(nl):
        kstack[i, 0] = z[f"patch_encoder.in_layer_caches_{i}_0"][0].astype(np.float16)
        vstack[i, 0] = z[f"patch_encoder.in_layer_caches_{i}_1"][0].astype(np.float16)
    state = {"keyCache": rt.NDArray(kstack), "valueCache": rt.NDArray(vstack)}
    r = await fn(inputs={
        "latent_patch": rt.NDArray(np.ascontiguousarray(latent.numpy())),
        "conv_tail": rt.NDArray(np.ascontiguousarray(conv_tail.numpy())),
        "pos": rt.NDArray(np.ascontiguousarray(np.array([pos0], np.int32))),
    }, state=state)
    c_emb = cos(r["embedding"].numpy(), z["patch_encoder.out_embedding"])
    # new_conv_tail is host-derived = the last latent frame of the input; verify it matches oracle
    host_tail = latent.numpy()[:, -1:, :].transpose(0, 2, 1)  # [1,4,128] -> last frame -> [1,128,1]
    c_tail = cos(host_tail, z["patch_encoder.out_conv_tail"])
    print(f"  engine emb cos={c_emb:.6f}   host-derived conv_tail cos={c_tail:.6f}")
    return min(c_emb, c_tail)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--buf", type=int, default=1000)
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    c = await run(a.src, a.buf, torch.float16)
    print(f"\n>>> patch_encoder export+engine: cos={c:.6f} -> {'GATE PASS' if c >= 0.99 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
