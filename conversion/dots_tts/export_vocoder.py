# Community port — NOT an Apple model.
"""Export the dots.tts vocoder (AudioVAE / BigVGAN 48 kHz decoder) to a Core AI .aimodel +
engine-gate vs the golden oracle wav. Unlike the qwen2/DiT stages (custom from-scratch graphs),
the vocoder is a standard op stack (Snake/SnakeBeta, ConvTranspose1d, grouped Conv1d) that the
coreai converter exports by TRACING the upstream torch module. weight_norm is folded at load
(remove_weight_norm); do_sample=False makes it deterministic (no VAE sampling).

  latents[1,128,Tf] -> inference_from_latents(do_sample=False) -> wav[1,1,1920*Tf]

  PYTHONPATH="<upstream-src>:<shims>:." <coreai-venv>/bin/python export_vocoder.py \
      --src <weights/dots.tts-soar> --upstream <upstream-src>

The upstream AudioVAE module imports only torch + local dots_tts submodules (no transformers),
so it constructs standalone inside the coreai venv.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

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


def build_vocoder(src: Path, DT):
    from safetensors.torch import load_file
    from dots_tts.modules.vocoder.bigvgan import AudioVAE
    from dots_tts.modules.vocoder.config import AudioVAEConfig
    cfg = AudioVAEConfig(**json.loads((src / "config.json").read_text())["vocoder"])
    av = AudioVAE(cfg).eval()
    av.remove_weight_norm()  # matches the saved artifact (saved post-remove_weight_norm)
    sd = load_file(str(src / "vocoder.safetensors"))
    missing, unexpected = av.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "weight_g" not in k and "weight_v" not in k]
    if missing or unexpected:
        print(f"  [warn] load: {len(missing)} missing e.g. {missing[:3]}, "
              f"{len(unexpected)} unexpected e.g. {unexpected[:3]}", flush=True)
    return av.to(DT).eval(), cfg


class VocWrap(nn.Module):
    def __init__(self, av):
        super().__init__(); self.av = av

    def forward(self, latents):
        return self.av.inference_from_latents(latents, do_sample=False)


async def run(src, DT, frames):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    src = Path(src)
    av, cfg = build_vocoder(src, DT)
    COL = 1920

    z = np.load(ART / "oracle_ref.npz")
    full = torch.from_numpy(z["vocoder.in_x"]).to(DT)          # [1,128,60]
    Tf = frames if frames else full.shape[-1]
    x = full[:, :, :Tf].contiguous()
    print(f"  vocoder latents {tuple(x.shape)} (Tf={Tf})", flush=True)

    wrap = VocWrap(av).eval()
    with torch.inference_mode():
        t_wav = wrap(x)
        # causality check: does the Tf-column chunk == the first Tf*1920 samples of the full vocode?
        if Tf < full.shape[-1]:
            t_full = wrap(full)
            c_causal = cos(t_wav.numpy().reshape(-1)[:Tf * COL], t_full.numpy().reshape(-1)[:Tf * COL])
            print(f"  causality: chunk vs full-prefix cos={c_causal:.6f} "
                  f"({'CLEAN (causal, streaming-safe)' if c_causal > 0.999 else 'boundary dependency'})", flush=True)

    ref = {"latents": x}
    prog = export_to_coreai(wrap, ref, dynamic_shapes=None,
                            input_names=("latents",), output_names=("wav",), state_names=None)
    ddir = ART / f"dots_vocoder_fp16_t{Tf}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir.name} ({_du(aim)})", flush=True)

    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(x.numpy()))})
    wav = r["wav"].numpy()
    print(f"  engine wav shape={wav.shape} peak={np.abs(wav).max():.3f}", flush=True)
    c = cos(wav.reshape(-1), t_wav.numpy().reshape(-1))
    print(f"  engine vs torch (this Tf) cos={c:.6f}")
    return c


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--upstream", required=True, help="path to dots.tts/src")
    ap.add_argument("--shims", default=None)
    ap.add_argument("--frames", type=int, default=0, help="Tf (latent columns); 0=oracle 60. 8=streaming (2 patches)")
    a = ap.parse_args()
    sys.path.insert(0, a.upstream)
    if a.shims:
        sys.path.insert(0, a.shims)
    ART.mkdir(parents=True, exist_ok=True)
    c = await run(a.src, torch.float16, a.frames)
    print(f"\n>>> vocoder export+engine: cos={c:.6f} -> {'GATE PASS' if c >= 0.99 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
