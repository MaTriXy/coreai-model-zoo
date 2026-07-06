"""End-to-end gate of the SHIPPED artifact: drive the full streaming diarization host loop with the
exported Core AI .aimodel (Mac GPU) instead of eager torch, and compare total_preds to NeMo's
forward_streaming. This validates the actual fixed-buffer graph + host contract together — the real
proof that "Streaming Sortformer diarizes correctly on Core AI".

Run (MAIN coreai-models venv; _GPU_LOCK for the engine):
    coreai-models/.venv/bin/python gate_e2e_engine.py
"""
from __future__ import annotations
import asyncio
import os
import numpy as np
import torch
import torch.nn.functional as F
import coreai.runtime as rt
from host_loop import run, N_SPK
from export_sortformer import TF_MAX, SPK, PE_MAX, T, MEL

HERE = os.path.dirname(os.path.abspath(__file__))
AIMODEL = os.path.join(HERE, "artifacts", "sortformer_float16.aimodel")


def pre_encode_len(tf: int) -> int:
    """dw_striding output length: 3 stages of (l-1)//2 + 1 (k=3, stride=2, pad=1)."""
    l = tf
    for _ in range(3):
        l = (l - 1) // 2 + 1
    return l


def engine_forward(fn, loop, dtype=np.float16):
    def f(chunk_feat, spkcache):
        tf = chunk_feat.shape[1]
        pe_len = pre_encode_len(tf)
        spk_len = spkcache.shape[1]

        chunk_mel = np.zeros((1, TF_MAX, MEL), dtype=dtype)
        chunk_mel[:, :tf] = chunk_feat.numpy().astype(dtype)
        spk188 = np.zeros((1, SPK, 512), dtype=dtype)
        spk188[:, :spk_len] = spkcache.numpy().astype(dtype)
        valid = np.zeros((1, T), dtype=dtype)
        valid[:, :spk_len] = 1.0
        valid[:, SPK : SPK + pe_len] = 1.0

        res = loop.run_until_complete(fn({
            "chunk_mel": rt.NDArray(chunk_mel),
            "spkcache": rt.NDArray(spk188),
            "valid": rt.NDArray(valid),
        }))
        preds_fixed = torch.from_numpy(res["preds"].numpy().astype(np.float32))[0]      # [378,4]
        chunk_pe_fixed = torch.from_numpy(res["chunk_pe"].numpy().astype(np.float32))[0]  # [190,512]

        # reconstruct [spkcache | chunk] concat-layout preds + valid chunk_pe for streaming_update
        preds_concat = torch.cat([preds_fixed[:spk_len], preds_fixed[SPK : SPK + pe_len]], dim=0)
        chunk_pe = chunk_pe_fixed[:pe_len]
        return preds_concat.unsqueeze(0), chunk_pe.unsqueeze(0)
    return f


def main():
    d = np.load(os.path.join(HERE, "stream_ref.npz"))
    mel = torch.from_numpy(d["mel"]).float()
    mel_len = torch.from_numpy(d["mel_len"]).long()
    ref = torch.from_numpy(d["total_preds"]).float()

    loop = asyncio.new_event_loop()
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = loop.run_until_complete(rt.AIModel.load(AIMODEL, opts))
    fn = m.load_function("main")
    print("[engine] loaded", os.path.basename(AIMODEL), "on GPU")

    total = run(engine_forward(fn, loop), mel, mel_len)
    n = min(total.shape[1], ref.shape[1])
    got, exp = total[:, :n], ref[:, :n]
    cos = F.cosine_similarity(got.reshape(-1), exp.reshape(-1), dim=0).item()
    maxd = (got - exp).abs().max().item()
    agree = ((got > 0.5) == (exp > 0.5)).float().mean().item()
    print(f"total_preds engine {tuple(total.shape)} vs NeMo ref {tuple(ref.shape)}")
    print(f"cos {cos:.6f}  max|Δ| {maxd:.5f}  activity-agree {agree*100:.2f}%")
    ok = agree > 0.99 and total.shape[1] == ref.shape[1]
    print("->", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
