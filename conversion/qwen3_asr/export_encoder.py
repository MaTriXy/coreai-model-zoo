"""P2: export the AuT audio encoder to a static fp16 Core AI `.aimodel` + self-gate vs golden.

Mirrors export_minicpmv46_vision.py. Bakes K chunks: input_features [1,128,100*K] + attn_bias
[1,S,S] (S=K*13) -> audio_embeds [S,2048]; host trims to the clip's N. Gates on the ja1 oracle
(K=5, N=55) against oracle_tokens.npz['encoder_out'] (fp32).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration  # noqa: E402
from audio_encoder import AuTEncoderStatic, build_attn_bias  # noqa: E402

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
DTYPE = torch.float16
OUTDIR = Path(__file__).resolve().parent
ART = OUTDIR / "artifacts"


async def gate(aimodel: Path, mel_fp16: np.ndarray, bias_fp16: np.ndarray,
               golden: torch.Tensor, N: int) -> bool:
    print(f"[gate] loading {aimodel.name} on GPU ...", flush=True)
    m = await rt.AIModel.load(
        str(aimodel),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    res = await asyncio.wait_for(fn(inputs={
        "input_features": rt.NDArray(np.ascontiguousarray(mel_fp16)),
        "attn_bias": rt.NDArray(np.ascontiguousarray(bias_fp16)),
    }), timeout=600)
    out = torch.from_numpy(res["audio_embeds"].numpy().astype(np.float32))[:N]
    cos = torch.nn.functional.cosine_similarity(out.reshape(-1), golden.reshape(-1), dim=0).item()
    pertok = torch.nn.functional.cosine_similarity(out, golden, dim=-1)
    maxabs = (out - golden).abs().max().item()
    print(f"[gate] engine out {tuple(out.shape)}  global cos {cos:.6f}  "
          f"per-token cos mean {pertok.mean():.6f} min {pertok.min():.6f}  max|Δ| {maxabs:.4f}", flush=True)
    return pertok.mean().item() > 0.999 and pertok.min().item() > 0.99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=5, help="K chunks baked into the bundle")
    args = ap.parse_args()
    K = args.chunks

    print(f"loading audio_tower (fp16, K={K}) ...", flush=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=DTYPE).eval()
    tower = model.thinker.audio_tower
    enc = AuTEncoderStatic(tower, K).eval()
    S = K * enc.tok_per_chunk
    chunk_mel = enc.chunk_mel

    # golden + real mel for the gate (ja1)
    d = np.load(OUTDIR / "oracle_tokens.npz")
    golden = torch.from_numpy(d["encoder_out"]).float()
    N = golden.shape[0]
    feats = torch.from_numpy(d["input_features"]).float()
    mel = torch.nn.functional.pad(feats, (0, K * chunk_mel - feats.shape[-1])).to(DTYPE)
    bias = build_attn_bias(S, N, window=enc.window, dtype=DTYPE)

    with torch.no_grad():
        eager = enc(mel, bias)[:N].float()
    c = torch.nn.functional.cosine_similarity(eager, golden, dim=-1).mean().item()
    print(f"[eager fp16] cos vs fp32 golden {c:.6f}", flush=True)

    example = {
        "input_features": torch.zeros(1, tower.num_mel_bins, K * chunk_mel, dtype=DTYPE),
        "attn_bias": torch.zeros(1, S, S, dtype=DTYPE),
    }
    print("[export] AuT encoder -> Core AI ...", flush=True)
    prog = export_to_coreai(
        enc, example, dynamic_shapes=None,
        input_names=("input_features", "attn_bias"), output_names=("audio_embeds",),
        state_names=None, externalize_modules=[])
    prog.optimize()

    ART.mkdir(parents=True, exist_ok=True)
    aimodel = ART / f"qwen3_asr_1.7b_audio_encoder_fp16_k{K}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"[save] {aimodel}  (emits [{S}, 2048])", flush=True)

    ok = asyncio.run(gate(aimodel, mel.numpy(), bias.numpy(), golden, N))
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'} — AuT encoder .aimodel "
          f"{'matches' if ok else 'DIVERGES from'} golden", flush=True)


if __name__ == "__main__":
    main()
