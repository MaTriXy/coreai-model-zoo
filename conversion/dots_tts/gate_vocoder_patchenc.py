"""Gate the vocoder (BigVGAN/AudioVAE) + patch_encoder standalone vs the oracle
(stages e + c of the ladder).

Unlike the qwen2/DiT stages (from-scratch reimplementations of the *custom* export
graph), the vocoder and patch_encoder are standard op stacks the coreai converter
exports by TRACING the torch module (weight_norm already folded at load; Snake/
ConvTranspose1d/LSTM are known-good zoo ops). The ladder step that matters for them is
therefore ISOLATION: drive each submodule standalone from the exact captured fixtures
and confirm it reproduces the oracle output deterministically — proving the export unit
can be lifted out of the generate() loop.

  * vocoder:      inference_from_latents(vocoder.in_x, do_sample=False) == vocoder.out_wav == wav
  * patch_encoder: decode_patch(latent_patch, conv_tail, layer_caches, positions) == out_embedding

Run (repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  PYTHONPATH="$W/_shims:$W/dots.tts/src" $W/venv/bin/python \
      conversion/dots_tts/gate_vocoder_patchenc.py --src $W/weights/dots.tts-soar --artifacts conversion/dots_tts/artifacts
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--artifacts", default="conversion/dots_tts/artifacts")
    ap.add_argument("--npz", default="oracle_ref.npz")
    args = ap.parse_args()

    z = np.load(Path(args.artifacts) / args.npz)
    from dots_tts.runtime import DotsTtsRuntime
    rt = DotsTtsRuntime.from_pretrained(args.src, precision="float32")
    model = rt.model
    ok = True

    # ---- vocoder ----
    x = torch.from_numpy(z["vocoder.in_x"]).to(torch.float32)
    with torch.no_grad():
        wav = model.vocoder.inference_from_latents(x, do_sample=False)
    c_voc = cos(wav.numpy(), z["vocoder.out_wav"])
    c_wav = cos(wav.numpy().reshape(-1), z["wav"])
    print("=== GATE e: vocoder (AudioVAE/BigVGAN, do_sample=False) ===")
    print(f"  in {tuple(x.shape)} -> wav {tuple(wav.shape)}  cos(vs out_wav)={c_voc:.6f}  cos(vs golden wav)={c_wav:.6f}")
    ok = (c_voc >= 0.999 and c_wav >= 0.999) and ok
    print("  ", "PASS ✅" if (c_voc >= 0.999 and c_wav >= 0.999) else "FAIL ❌")

    # ---- patch_encoder.decode_patch ----
    pe = model.core.patch_encoder
    latent_patch = torch.from_numpy(z["patch_encoder.in_latent_patch"]).to(torch.float32)
    conv_tail = torch.from_numpy(z["patch_encoder.in_conv_tail"]).to(torch.float32)
    positions = torch.from_numpy(z["patch_encoder.in_positions"]).to(torch.long)
    n_layers = len([k for k in z.files if k.startswith("patch_encoder.in_layer_caches_") and k.endswith("_0")])
    layer_caches = tuple(
        (torch.from_numpy(z[f"patch_encoder.in_layer_caches_{i}_0"]).to(torch.float32),
         torch.from_numpy(z[f"patch_encoder.in_layer_caches_{i}_1"]).to(torch.float32))
        for i in range(n_layers)
    )
    with torch.no_grad():
        emb, new_tail = pe.decode_patch(latent_patch, conv_tail, layer_caches, positions)
    c_emb = cos(emb.numpy(), z["patch_encoder.out_embedding"])
    c_tail = cos(new_tail.numpy(), z["patch_encoder.out_conv_tail"])
    print("=== GATE c: patch_encoder.decode_patch (streaming causal) ===")
    print(f"  latent{tuple(latent_patch.shape)} + {n_layers}L caches -> emb {tuple(emb.shape)}  "
          f"cos(emb)={c_emb:.6f}  cos(tail)={c_tail:.6f}")
    ok = (c_emb >= 0.999 and c_tail >= 0.999) and ok
    print("  ", "PASS ✅" if (c_emb >= 0.999 and c_tail >= 0.999) else "FAIL ❌")

    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
