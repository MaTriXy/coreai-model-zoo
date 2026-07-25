# Community port — NOT an Apple model.
"""Export dots.tts host-side glue weights (the projections run on the CPU outside the engine
bundles) to artifacts/dots_host_glue/{manifest.json, <name>.bin}, mirroring
voxcpm/export_host_glue_v2.py. Loaded by DotsGlue.swift.

  embed_tokens    fp16 [vocab,1536]   token embedding (prefill text lookup)  = llm.model.embed_tokens
  hidden_proj_{w,b} fp32 [1024,1536]/[1024]   fm hidden (append_hidden_chunk)
  latent_proj_{w,b} fp32 [1024,128]/[1024]    fm history (append_history_chunk, NORMALIZED patch)
  coordinate_proj_{w,b} fp32 [1024,128]/[1024]  noise -> DiT space (solver scatter)
  eos_proj0_{w,b}  fp32 [1536,1536]/[1536]  +  eos_proj2_{w,b} fp32 [2,1536]/[2]  (SiLU between)
  latent_mean, latent_std  fp32 [128]     denormalize = x*std + mean (latent_stats.pt: mean, var)

  PYTHONPATH=. <coreai-venv>/bin/python export_host_glue.py --src <weights/dots.tts-soar>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ART = Path(__file__).resolve().parent / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    sd = load_file(str(src / "model.safetensors"))
    out = ART / "dots_host_glue"
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    def dump(name, arr, dtype):
        arr = np.ascontiguousarray(arr.astype(dtype))
        (out / f"{name}.bin").write_bytes(arr.tobytes())
        manifest[name] = {"dtype": {np.float16: "fp16", np.float32: "fp32"}[dtype],
                          "shape": list(arr.shape), "bytes": arr.nbytes}

    def g(k):
        return sd[k].to(torch.float32).numpy()

    dump("embed_tokens", g("llm.model.embed_tokens.weight"), np.float16)
    for name, key in [("hidden_proj", "hidden_proj"), ("latent_proj", "latent_proj"),
                      ("coordinate_proj", "coordinate_proj")]:
        dump(f"{name}_w", g(f"{key}.weight"), np.float32)
        dump(f"{name}_b", g(f"{key}.bias"), np.float32)
    dump("eos_proj0_w", g("eos_proj.0.weight"), np.float32)
    dump("eos_proj0_b", g("eos_proj.0.bias"), np.float32)
    dump("eos_proj2_w", g("eos_proj.2.weight"), np.float32)
    dump("eos_proj2_b", g("eos_proj.2.bias"), np.float32)

    st = torch.load(str(src / "latent_stats.pt"), weights_only=False)
    dump("latent_mean", torch.as_tensor(st["mean"]).float().numpy(), np.float32)
    dump("latent_std", torch.sqrt(torch.as_tensor(st["var"]).float()).numpy(), np.float32)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"-> {out}  ({len(manifest)} tensors, {total/1e6:.1f} MB)")
    for k, v in manifest.items():
        print(f"  {k:20s} {v['dtype']:4s} {v['shape']}")


if __name__ == "__main__":
    main()
