"""Gate the Qwen2.5 backbone overlay vs the torch oracle (stage b of the ladder).

Loads `llm.kw_inputs_embeds` from oracle_ref.npz, runs Qwen2Backbone (clean torch,
baked RoPE, explicit GQA) loaded from the upstream model.safetensors, and asserts
cosine >= 0.999 on both the last-layer hidden (vs llm.out1) and the tied logits
(vs llm.out0).

Run (repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  $W/venv/bin/python conversion/dots_tts/gate_llm.py \
      --src $W/weights/dots.tts-soar --artifacts conversion/dots_tts/artifacts
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from torch_overlays import Qwen2Backbone


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--artifacts", default="conversion/dots_tts/artifacts")
    ap.add_argument("--npz", default="oracle_ref.npz")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = json.loads((src / "llm_config.json").read_text())
    z = np.load(Path(args.artifacts) / args.npz)

    sd = load_file(str(src / "model.safetensors"))
    sd = {k: v for k, v in sd.items() if k.startswith("llm.")}

    model = Qwen2Backbone(cfg).to(torch.float32).eval()
    model.load_upstream(sd)

    emb = torch.from_numpy(z["llm.kw_inputs_embeds"]).to(torch.float32)
    with torch.no_grad():
        hidden, logits = model(emb)

    c_hidden = cos(hidden.numpy(), z["llm.out1"])
    c_logits = cos(logits.numpy(), z["llm.out0"])
    # also max-abs err on hidden
    mae_hidden = float(np.abs(hidden.numpy() - z["llm.out1"]).max())
    # argmax agreement on logits (the sampling-relevant metric)
    am_ov = logits.numpy()[0].argmax(-1)
    am_or = z["llm.out0"][0].argmax(-1)
    am_agree = float((am_ov == am_or).mean())

    print(f"=== GATE b: Qwen2.5 backbone ===")
    print(f"  inputs_embeds {tuple(emb.shape)} -> hidden {tuple(hidden.shape)} logits {tuple(logits.shape)}")
    print(f"  cos(hidden vs llm.out1) = {c_hidden:.6f}   max|err| = {mae_hidden:.3e}")
    print(f"  cos(logits vs llm.out0) = {c_logits:.6f}   argmax agree = {am_agree:.3f}")
    ok = c_hidden >= 0.999 and c_logits >= 0.999
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
