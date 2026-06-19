"""Export the fixed-shape Qwen2.5-Omni audio encoder to Core AI (fp16).

Follows the whisper recipe (coreai-models/models/whisper/export.py): a single
STATIC-shape torch.export -> TorchConverter().to_coreai().  No KV state, no
externalize, no dynamic dims -> one compiled graph (ANE-ready).

Bundle contract:
    input_features [1, 128, 200*K] fp16   (host zero-pads mel to K full chunks)
    attn_bias      [K, 1, 1, 100]  fp16   (0 valid post-CNN frame / -30000 pad)
        -> audio_embeds [K*50, 2048] fp16 (host trims to the clip's real N)

K (n_chunks) is baked per bundle; K=2 matches the N=100 decoder bundle / 4 s clip.

Run: coreai-models/.venv/bin/python ondevice/export_qwen2_5_omni_audio.py [--chunks K]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from coreai_torch import TorchConverter, get_decomp_table

import coreai.runtime as rt
from coreai_models.models.macos.qwen2_5_omni_audio import Qwen2_5OmniAudioEncoderStatic

HF_ID = "Qwen/Qwen2.5-Omni-3B"
ART = Path(__file__).resolve().parent / "artifacts"
DTYPE = torch.float16
CHUNK_MEL = 200
HEADS_FRAMES = 100   # post-CNN frames per chunk (n_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2, help="K: fixed chunk count (mel width = 200*K)")
    args = ap.parse_args()
    K = args.chunks

    print(f"loading audio tower (fp16, K={K}) from {HF_ID} ...", flush=True)
    enc = Qwen2_5OmniAudioEncoderStatic.from_hf(HF_ID, K, target_dtype=DTYPE).eval()

    example = {
        "input_features": torch.zeros(1, enc.tower.num_mel_bins, CHUNK_MEL * K, dtype=DTYPE),
        "attn_bias": torch.zeros(K, 1, 1, HEADS_FRAMES, dtype=DTYPE),
    }

    print("torch.export (static) ...", flush=True)
    with torch.autocast(device_type="cpu", dtype=DTYPE):
        exported = torch.export.export(enc, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())

    print("converting to Core AI ...", flush=True)
    prog = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_features", "attn_bias"],
        output_names=["audio_embeds"],
    ).to_coreai()
    prog.optimize()

    out = ART / f"qwen2_5_omni_3b_audio_encoder_fp16_k{K}.aimodel"
    ART.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out, rt.AIModelAssetMetadata())
    sz = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"SAVED {out} ({sz})  -> emits [{K * (HEADS_FRAMES // 2)}, 2048] audio tokens")


if __name__ == "__main__":
    main()
