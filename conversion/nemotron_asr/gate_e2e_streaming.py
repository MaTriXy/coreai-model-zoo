"""Phase 7 — full STREAMING pipeline gate: mel chunks -> pre/pre_first + conformer .aimodels
(explicit caches) -> incremental host RNNT greedy loop driving predict/joint .aimodels ->
transcript, compared token-for-token to the HF streaming oracle.

The decode is interleaved exactly as the app will run it: after each chunk contributes its
4 encoder frames, the transducer consumes every newly available frame before the next chunk
(same tokens as offline decoding — greedy RNNT never looks ahead). Per-chunk wall time
(encode + decode) is reported against the 320 ms audio budget.

    coreai-models/.venv/bin/python gate_e2e_streaming.py [oracle_stream_en_US.npz]
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch

import coreai.runtime as rt

from export_encoder_streaming import CHUNK_FIRST, CONV_K, HDIM, HEADS, HID, KV, Q, SPLIT, build_neg_mask, mel_chunks

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DEC_HID, DEC_NL, BLANK, MAX_SYMS = 640, 2, 13087, 10


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def main():
    oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle_stream_en_US.npz"
    d = np.load(HERE / oracle)
    mel = torch.from_numpy(d["mel"]).float()
    one_hot = torch.from_numpy(d["one_hot"]).float()[None]
    gold_tokens = d["tokens"].tolist()
    text = str(d["text"])
    T = int(d["T"])

    pf = await gpu(ART / "nemotron_asr_stream_pre_first_float16.aimodel")
    pr = await gpu(ART / "nemotron_asr_stream_pre_float16.aimodel")
    ca = await gpu(ART / "nemotron_asr_stream_conformer_a_float16.aimodel")
    cb = await gpu(ART / "nemotron_asr_stream_conformer_b_float16.aimodel")
    pd = await gpu(ART / "nemotron_asr_predict_float32.aimodel")
    jt = await gpu(ART / "nemotron_asr_joint_float32.aimodel")

    nd16 = lambda t: rt.NDArray(t.half().numpy())
    z16 = lambda *s: rt.NDArray(np.zeros(s, dtype=np.float16))

    async def step_p(token, h, c):
        o = await pd({"token": rt.NDArray(np.array([[token]], dtype=np.int32)),
                      "h": rt.NDArray(h), "c": rt.NDArray(c)})
        return o["dec_out"].numpy(), o["h_out"].numpy(), o["c_out"].numpy()

    async def step_j(dec, enc_frame):
        o = await jt({"dec_out": rt.NDArray(dec), "enc_frame": rt.NDArray(enc_frame)})
        return o["token_logits"].numpy()

    kA, vA = z16(SPLIT, HEADS, KV - Q, HDIM), z16(SPLIT, HEADS, KV - Q, HDIM)
    ccA = z16(SPLIT, HID, CONV_K - 1)
    kB, vB = z16(SPLIT, HEADS, KV - Q, HDIM), z16(SPLIT, HEADS, KV - Q, HDIM)
    ccB = z16(SPLIT, HID, CONV_K - 1)
    c0 = c1 = c2 = None
    h = np.zeros((DEC_NL, 1, DEC_HID), dtype=np.float32)
    c = np.zeros((DEC_NL, 1, DEC_HID), dtype=np.float32)
    dec, h, c = await step_p(BLANK, h, c)

    enc_buf: list[np.ndarray] = []
    frame, syms, steps, emitted = 0, 0, 0, []
    dt_ms = []
    for i, chunk in enumerate(mel_chunks(mel)):
        t0 = time.perf_counter()
        if i == 0:
            r = await pf({"mel": nd16(chunk)})
            e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0", "cache1", "cache2"))
        else:
            r = await pr({"mel": nd16(chunk), "cache0": c0, "cache1": c1, "cache2": c2})
            e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0_out", "cache1_out", "cache2_out"))
        neg = nd16(build_neg_mask(i))
        r = await ca({"embeds": e, "neg_mask": neg, "k_cache": kA, "v_cache": vA, "conv_cache": ccA})
        xh, kA, vA, ccA = r["embeds_out"], r["k_out"], r["v_out"], r["conv_out"]
        r = await cb({"embeds": xh, "one_hot": nd16(one_hot), "neg_mask": neg,
                      "k_cache": kB, "v_cache": vB, "conv_cache": ccB})
        kB, vB, ccB = r["k_out"], r["v_out"], r["conv_out"]
        enc_frames = r["enc_proj"].numpy().astype(np.float32)[0]        # [Q,640]
        enc_buf.extend(enc_frames)
        while frame < len(enc_buf) and steps < MAX_SYMS * T + 16:
            logits = await step_j(dec, enc_buf[frame][None])
            token = int(logits.argmax())
            steps += 1
            if token == BLANK:
                frame += 1; syms = 0
            else:
                emitted.append(token)
                dec, h, c = await step_p(token, h, c)
                syms += 1
                if syms >= MAX_SYMS:
                    frame += 1; syms = 0
        dt_ms.append((time.perf_counter() - t0) * 1e3)

    exact = emitted == gold_tokens
    same = sum(int(a == b) for a, b in zip(emitted, gold_tokens))
    budget = CHUNK_FIRST * 10  # ms of audio per steady chunk ~ 320
    print(f"[e2e streaming gpu] {len(dt_ms)} chunks  emitted {len(emitted)} (gold {len(gold_tokens)})  "
          f"exact={exact}  token-agree {same}/{len(gold_tokens)}")
    print(f"  per-chunk encode+decode: avg {np.mean(dt_ms[1:]):.1f}ms  max {np.max(dt_ms[1:]):.1f}ms  "
          f"(audio budget 320ms/chunk, RTF {np.mean(dt_ms[1:]) / 320:.3f})")
    print(f"  golden: {text!r}")
    print(f"\n{'✅ PASS — Nemotron 3.5 ASR STREAMS end-to-end on Core AI' if exact else '⚠️ mismatch — inspect near-ties'}")


if __name__ == "__main__":
    asyncio.run(main())
