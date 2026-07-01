# Community port — NOT an Apple model.
"""Eager parity of the STATIC prefill+decode modules vs the golden gen_ids (no export).

Drives Qwen3ASRPrefillStatic (seed cache, step-0 logits) + Qwen3ASRDecodeStatic (greedy loop on
scalar pos, shared cache) in plain torch fp16 to confirm the static cache/mask/RoPE math is correct
BEFORE the slow Core AI export. Uses the real encoder_out from oracle_tokens.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen3_asr_decoder import Qwen3ASRDecoderPipelined
from qwen3_asr_static import Qwen3ASRStaticDecode, Qwen3ASRStaticPrefill, build_kv_state

MODEL = "Qwen/Qwen3-ASR-1.7B"
DTYPE = torch.float16
AUDIO_TOKEN_ID = 151676
V = 151936
EOS = {151643, 151645}
CACHE_LEN = 256
OUTDIR = Path(__file__).resolve().parent


@torch.no_grad()
def main() -> None:
    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()[0]          # [70]
    audio = torch.from_numpy(d["encoder_out"]).to(DTYPE)      # [55, 2048]
    golden = d["gen_ids"].tolist()
    N = int(audio.shape[0])

    base = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=DTYPE)
    prefill = Qwen3ASRStaticPrefill(base).eval()
    decode = Qwen3ASRStaticDecode(base).eval()

    # rewrite the prompt's audio placeholders to V + slot
    prompt = ids.clone()
    aud_pos = (ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    prompt[aud_pos] = V + torch.arange(N)
    Sp = prompt.shape[0]

    st = build_kv_state(base.config, CACHE_LEN, DTYPE)
    kc, vc = st["k_cache"], st["v_cache"]

    # --- prefill: seed cache [0,Sp), get step-0 logits ---
    logits0 = prefill(prompt.unsqueeze(0).to(torch.int32), audio, kc, vc)  # [1,1,V]
    nxt = int(logits0[0, -1].float().argmax())

    # --- decode: greedy loop on scalar pos, shared cache ---
    gen = []
    p = Sp
    for _ in range(40):
        gen.append(nxt)
        if nxt in EOS:
            break
        iid = torch.tensor([[nxt]], dtype=torch.int32)
        pos = torch.tensor([p], dtype=torch.int32)
        logits = decode(iid, pos, kc, vc)  # [1,1,V]
        p += 1
        nxt = int(logits[0, -1].float().argmax())

    print(f"mine  ({len(gen)}): {gen}")
    print(f"golden({len(golden)}): {golden}")
    match = gen[: len(golden)] == golden
    print("\n=== STATIC EAGER PARITY ===")
    print("PASS — token-for-token match" if match else "FAIL")
    raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
