"""P3 capstone: clean full ON-ENGINE end-to-end gate with the STATIC bundles.

encoder .aimodel -> audio_embeds -> static PREFILL .aimodel (q_len=Sp, seeds the shared KV state,
returns step-0 logits) -> static DECODE .aimodel (q=1, scalar ``pos``, flat) greedy loop -> compare
to golden gen_ids. The keyCache/valueCache NDArray buffers are created ONCE and passed to both the
prefill and the decode function, so the engine threads the KV cache across the two bundles host-side.

This driver IS the host pipeline / CoreAIKit Transcriber logic. Run on the community venv.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_encoder import build_attn_bias  # noqa: E402

OUTDIR = Path(__file__).resolve().parent
ART = OUTDIR / "artifacts"
AUDIO_TOKEN_ID = 151676
V = 151936
EOS = {151643, 151645}
NLAYER, NKV, HD = 28, 8, 128


async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8hu")
    ap.add_argument("--cache-len", type=int, default=1024)
    ap.add_argument("--enc-chunks", type=int, default=30, help="encoder K (30 = ≤30s ship)")
    ap.add_argument("--unified", action="store_true", default=True,
                    help="load ONE unified bundle (prefill+decode entrypoints)")
    a = ap.parse_args()
    CACHE_LEN = a.cache_len
    ENC_AIM = ART / f"qwen3_asr_1.7b_audio_encoder_fp16_k{a.enc_chunks}.aimodel"

    import coreai.runtime as rt

    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()[0]              # [70]
    feats = torch.from_numpy(d["input_features"]).float()         # [1,128,417]
    golden = d["gen_ids"].tolist()
    N = int(d["encoder_out"].shape[0])                            # 55
    Sp = ids.shape[0]                                            # 70
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    PREF = ART / f"qwen3_asr_1.7b_{a.mode}_prefill_sp{Sp}"
    DEC = ART / f"qwen3_asr_1.7b_{a.mode}_decode_cl{CACHE_LEN}"

    # --- 1) encoder .aimodel -> audio_embeds [N, 2048] (host zero-pads mel to K chunks, trims to N) ---
    K = a.enc_chunks
    mel = torch.nn.functional.pad(feats, (0, K * 100 - feats.shape[-1])).to(torch.float16)
    bias = build_attn_bias(K * 13, N, window=104, dtype=torch.float16)
    enc = await rt.AIModel.load(str(ENC_AIM), gpu)
    efn = enc.load_function("main")
    eres = await asyncio.wait_for(efn(inputs={
        "input_features": rt.NDArray(np.ascontiguousarray(mel.numpy())),
        "attn_bias": rt.NDArray(np.ascontiguousarray(bias.numpy())),
    }), timeout=600)
    audio_embeds = eres["audio_embeds"].numpy().astype(np.float16)[:N]
    print(f"[enc] audio_embeds {audio_embeds.shape}", flush=True)
    del enc, efn

    # --- shared KV state (one set of buffers threaded through both bundles) ---
    state = {
        "keyCache": rt.NDArray(np.zeros((NLAYER, 1, NKV, CACHE_LEN, HD), dtype=np.float16)),
        "valueCache": rt.NDArray(np.zeros((NLAYER, 1, NKV, CACHE_LEN, HD), dtype=np.float16)),
    }

    # rewrite the 70-token prompt: audio positions -> V + slot
    prompt = ids.clone()
    aud_pos = (ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    prompt[aud_pos] = V + torch.arange(N)
    prompt_np = prompt.numpy().astype(np.int32)[None, :]          # [1, Sp]

    # --- load: one unified bundle (prefill+decode fns) or two single-fn bundles ---
    if a.unified:
        UNI = ART / f"qwen3_asr_1.7b_{a.mode}_unified_cl{CACHE_LEN}"
        um = await rt.AIModel.load(str(UNI / f"{UNI.name}.aimodel"), gpu)
        pfn, dfn = um.load_function("prefill"), um.load_function("decode")
    else:
        pm = await rt.AIModel.load(str(PREF / f"{PREF.name}.aimodel"), gpu)
        pfn = pm.load_function("main")

    # --- 2) static PREFILL: seed cache [0,Sp), step-0 logits ---
    t0 = time.time()
    pres = await asyncio.wait_for(pfn(inputs={
        "input_ids": rt.NDArray(np.ascontiguousarray(prompt_np)),
        "audio_embeds": rt.NDArray(np.ascontiguousarray(audio_embeds)),
    }, state=state), timeout=600)
    pref_ms = (time.time() - t0) * 1000.0
    logits = pres["logits"].numpy().astype(np.float32).reshape(-1)
    print(f"[prefill] {pref_ms:.0f} ms (q_len={Sp}); cache seeded", flush=True)

    # --- 3) static DECODE: greedy loop on scalar pos, shared cache ---
    if not a.unified:
        del pm, pfn
        dm = await rt.AIModel.load(str(DEC / f"{DEC.name}.aimodel"), gpu)
        dfn = dm.load_function("main")

    async def step(tok_id: int, p: int) -> np.ndarray:
        res = await asyncio.wait_for(dfn(inputs={
            "input_ids": rt.NDArray(np.ascontiguousarray(np.array([[tok_id]], dtype=np.int32))),
            "pos": rt.NDArray(np.ascontiguousarray(np.array([p], dtype=np.int32))),
        }, state=state), timeout=600)
        return res["logits"].numpy().astype(np.float32).reshape(-1)

    gen = []
    gaps = []
    nxt = int(np.argmax(logits))
    p = Sp
    for _ in range(40):
        gen.append(nxt)
        if nxt in EOS:
            break
        t0 = time.time()
        logits = await step(nxt, p)
        gaps.append((time.time() - t0) * 1000.0)
        p += 1
        nxt = int(np.argmax(logits))

    print(f"\nmine  ({len(gen)}): {gen}", flush=True)
    print(f"golden({len(golden)}): {golden}", flush=True)
    if gaps:
        print(f"[decode] gap min {min(gaps):.1f} / med {np.median(gaps):.1f} / "
              f"max {max(gaps):.1f} ms (n={len(gaps)})", flush=True)
    match = gen[:len(golden)] == golden
    print("\n=== CLEAN END-TO-END ON-ENGINE GATE (static bundles) ===")
    print("PASS — token-for-token match" if match else "FAIL", flush=True)
    raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    asyncio.run(main())
