"""P1 encoder parity: static AuTEncoderStatic vs the HF golden encoder_out.

Loads the official audio_tower weights, feeds the SAME mel (zero-padded to K full
chunks) the oracle used, and compares to oracle_tokens.npz['encoder_out'] (cos / max-abs).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

OFFICIAL = "/tmp/qwen3-asr-official"
sys.path.insert(0, OFFICIAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_encoder import AuTEncoderStatic, build_attn_bias  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
OUTDIR = Path(__file__).resolve().parent


@torch.no_grad()
def main() -> None:
    d = np.load(OUTDIR / "oracle_tokens.npz")
    input_features = torch.from_numpy(d["input_features"]).float()   # [1, 128, 417]
    golden = torch.from_numpy(d["encoder_out"]).float()              # [N, 2048]
    N = golden.shape[0]
    mel_frames = input_features.shape[-1]
    print(f"mel frames={mel_frames}  golden encoder_out={tuple(golden.shape)}  N={N}", flush=True)

    print("loading audio_tower ...", flush=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    tower = model.thinker.audio_tower
    chunk_mel = tower.n_window * 2                                   # 100
    K = math.ceil(mel_frames / chunk_mel)
    enc = AuTEncoderStatic(tower, K).eval()
    print(f"K={K}  chunk_mel={chunk_mel}  tok/chunk={enc.tok_per_chunk}  window={enc.window}", flush=True)

    # zero-pad mel to K*100 frames
    pad = K * chunk_mel - mel_frames
    mel = torch.nn.functional.pad(input_features, (0, pad))          # [1, 128, K*100]
    S = K * enc.tok_per_chunk
    bias = build_attn_bias(S, N, window=enc.window)

    out = enc(mel, bias)[:N]                                         # trim to valid N
    print(f"static out={tuple(out.shape)}", flush=True)

    cos = torch.nn.functional.cosine_similarity(out.reshape(-1), golden.reshape(-1), dim=0).item()
    maxabs = (out - golden).abs().max().item()
    relabs = ((out - golden).abs().mean() / golden.abs().mean()).item()
    print(f"\n=== ENCODER PARITY ===\ncos={cos:.8f}  max|Δ|={maxabs:.5e}  mean|Δ|/mean|g|={relabs:.5e}",
          flush=True)
    print("PASS" if cos > 0.9999 and maxabs < 1e-2 else "FAIL (investigate)", flush=True)


if __name__ == "__main__":
    main()
