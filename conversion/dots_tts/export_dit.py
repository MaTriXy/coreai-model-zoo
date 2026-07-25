# Community port — NOT an Apple model.
"""Export the dots.tts DiT flow head (velocity_field_predictor) to a Core AI .aimodel + engine-gate
vs the golden oracle. One unrolled denoise step (the euler/meanflow solver wraps it at the host):

  x[B,S,1024] + timesteps[B] + attn_mask[1,S,S] + pos_ids[1,S] + g_cond[B,1024] (+duration[B] mf)
      -> velocity[B,S,128]

Modes: soar (flow_matching, CFG batch-2) / mf (meanflow, batch-1, +duration). Kept fp16 (DiT +
vocoder are quant-sensitive continuous-feedback — MLX/VoxCPM lesson).

NOTE on shape: at inference the DiT attends over the GROWING fm history (S = fm_seq_len + 4;
patch 1 is S=5, later patches larger — model.py:_prepare_fm_decode_inputs, bucketed 32/64/../1024
patches → total_len = bucket*5 + 4). **AdaLN modulation is timestep-dependent, so the per-step
K/V change → NO cross-step KV cache; the DiT re-attends the full modulated sequence each solver
step.** The export is therefore BUCKETED fixed-shape: the DiTOverlay takes attn_mask/pos_ids as
runtime INPUTS, so ONE fixed-S bundle serves every fm_seq_len <= S-4 (host builds the mask/pos_ids).
This script exports at `--total-len S`:
  * S=5  -> golden gate vs the oracle dit.out0 (validates numerics).
  * S>5  -> a bucket bundle; self-consistency gate (engine vs the oracle-gated torch overlay) on a
            realistic host-built mask/pos_ids at fm_seq_len=S-4 (validates conversion at that shape).
Only the last 4 positions' velocity is used downstream (core.py: `vt = vt[:, latent_start:]`).

  PYTHONPATH=. <coreai-venv>/bin/python export_dit.py --mode soar --src <weights/dots.tts-soar>
  PYTHONPATH=. <coreai-venv>/bin/python export_dit.py --mode mf --src <..-mf> --total-len 164   # bucket 32
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
from torch_overlays import DiTOverlay  # noqa: E402

_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
ART = Path(__file__).resolve().parent / "artifacts"


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# --- host mask/pos_ids builders (verbatim from model.py:_build_fm_{attn_mask,pos_ids}) ---
def build_fm_attn_mask(fm_seq_len: int, total_len: int, latent_patch: int = 4, hidden_patch: int = 1):
    m = torch.zeros((1, total_len, total_len), dtype=torch.bool)
    latent_start = total_len - latent_patch
    block_start = fm_seq_len - hidden_patch
    if block_start > 0:
        m[:, :block_start, :block_start] = torch.ones(block_start, block_start, dtype=torch.bool).triu(1).logical_not()
    m[:, block_start:fm_seq_len, :fm_seq_len] = True
    m[:, block_start:fm_seq_len, latent_start:] = True
    m[:, latent_start:, :fm_seq_len] = True
    m[:, latent_start:, latent_start:] = True
    if latent_start > fm_seq_len:
        idx = torch.arange(fm_seq_len, latent_start)
        m[:, idx, idx] = True
    return m


def build_fm_pos_ids(fm_seq_len: int, total_len: int, latent_patch: int = 4):
    p = torch.zeros((1, total_len), dtype=torch.float32)
    latent_start = total_len - latent_patch
    p[:, :fm_seq_len] = torch.arange(fm_seq_len, dtype=torch.float32)
    p[:, latent_start:] = torch.arange(fm_seq_len, fm_seq_len + latent_patch, dtype=torch.float32)
    return p


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


class DiTWrapSoar(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, timesteps, attn_mask, pos_ids, g_cond):
        return self.m(x, timesteps, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g_cond)


class DiTWrapMf(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, timesteps, attn_mask, pos_ids, g_cond, duration):
        return self.m(x, timesteps, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g_cond, duration=duration)


def _load_overlay(src, mode, DT):
    dit_mode = "meanflow" if mode == "mf" else "flow_matching"
    cfg = json.loads((Path(src) / "config.json").read_text())["DiT"]
    sd = {k: v for k, v in load_file(str(Path(src) / "model.safetensors")).items()
          if k.startswith("velocity_field_predictor.")}
    m = DiTOverlay(cfg, mode=dit_mode).to(DT).eval()
    m.load_upstream(sd)
    return m


def _wrap_and_names(m, mode):
    if mode == "mf":
        return DiTWrapMf(m).eval(), ("x", "timesteps", "attn_mask", "pos_ids", "g_cond", "duration")
    return DiTWrapSoar(m).eval(), ("x", "timesteps", "attn_mask", "pos_ids", "g_cond")


def _oracle_ref(mode, DT):
    """S=5 golden inputs from the oracle fixture."""
    z = np.load(ART / ("oracle_ref_mf.npz" if mode == "mf" else "oracle_ref.npz"))

    def g(k, dt=DT):
        return torch.from_numpy(z[k]).to(dt)
    ref = {"x": g("dit.kw_x"), "timesteps": g("dit.kw_timesteps"), "attn_mask": g("dit.kw_attn_mask"),
           "pos_ids": g("dit.kw_pos_ids", torch.float32), "g_cond": g("dit.kw_g_cond")}
    if mode == "mf":
        ref["duration"] = g("dit.kw_duration")
    return ref, z["dit.out0"]


def _synthetic_ref(mode, total_len, DT):
    """A realistic host-built bucket state at fm_seq_len = total_len-4 (full bucket, no padding)."""
    B = 1 if mode == "mf" else 2
    fm_seq_len = total_len - 4
    torch.manual_seed(1234 + total_len)
    ref = {
        "x": torch.randn(B, total_len, 1024, dtype=DT),
        "timesteps": torch.rand(B, dtype=DT),
        "attn_mask": build_fm_attn_mask(fm_seq_len, total_len).to(DT),
        "pos_ids": build_fm_pos_ids(fm_seq_len, total_len),
        "g_cond": torch.randn(B, 1024, dtype=DT),
    }
    if mode == "mf":
        ref["duration"] = torch.rand(1, dtype=DT)
    return ref


async def run(mode, src, total_len, DT):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = _load_overlay(src, mode, DT)
    wrap, names = _wrap_and_names(m, mode)

    golden = None
    if total_len == 5:
        ref, golden = _oracle_ref(mode, DT)  # gate vs oracle dit.out0
    else:
        ref = _synthetic_ref(mode, total_len, DT)  # self-consistency vs torch overlay

    with torch.inference_mode():
        t_out = wrap(*[ref[n] for n in names]).numpy()
    if golden is not None:
        print(f"  torch overlay vs golden dit.out0: cos={cos(t_out, golden):.6f}", flush=True)

    prog = export_to_coreai(wrap, ref, dynamic_shapes=None,
                            input_names=names, output_names=("velocity",), state_names=None)
    ddir = ART / f"dots_dit_{mode}_fp16_s{total_len}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir.name} ({_du(aim)})", flush=True)

    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    inp = {n: rt.NDArray(np.ascontiguousarray(ref[n].numpy())) for n in names}
    r = await fn(inputs=inp)
    eng = r["velocity"].numpy()
    ref_out = golden if golden is not None else t_out
    tag = "golden dit.out0" if golden is not None else "torch overlay (self-consistency)"
    # only the last 4 positions' velocity is used downstream
    c_full = cos(eng, ref_out)
    c_tail = cos(eng[:, -4:], np.asarray(ref_out)[:, -4:])
    print(f"  engine velocity vs {tag}: cos(full)={c_full:.6f}  cos(last4=used)={c_tail:.6f}")
    return min(c_full, c_tail)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["soar", "mf"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--total-len", type=int, default=5,
                    help="DiT sequence length S. 5=oracle golden gate; bucket S=32/64/..*5+4 (164,324,644,..)")
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    c = await run(a.mode, a.src, a.total_len, torch.float16)
    print(f"\n>>> dit/{a.mode} s{a.total_len} export+engine: cos={c:.6f} -> {'GATE PASS' if c >= 0.99 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
