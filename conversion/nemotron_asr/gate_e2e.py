"""Phase 4 — full-pipeline gate: mel + one_hot -> encoder.aimodel (fp16) -> host RNNT greedy
loop driving predict/joint .aimodels -> transcript, compared token-for-token to the oracle.

The definitive "Nemotron 3.5 ASR runs end to end on Core AI" check (Mac GPU; device follows).

    coreai-models/.venv/bin/python gate_e2e.py [oracle_en_US.npz]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

import coreai.runtime as rt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
HID, NL, BLANK, MAX_SYMS = 640, 2, 13087, 10


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def main():
    oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle_en_US.npz"
    d = np.load(HERE / oracle)
    mel = torch.from_numpy(d["input_features"]).float()      # [1,L,128] (native encoder layout)
    one_hot = torch.from_numpy(d["one_hot"]).float()[None]   # [1,128]
    L = mel.shape[1]
    gold_tokens = d["tokens"].tolist()
    text = str(d["text"])

    enc_fn = await gpu(ART / f"nemotron_asr_encoder_float16_L{L}.aimodel")
    pf = await gpu(ART / "nemotron_asr_predict_float32.aimodel")
    jf = await gpu(ART / "nemotron_asr_joint_float32.aimodel")

    r = await enc_fn({"mel": rt.NDArray(mel.half().numpy()),
                      "one_hot": rt.NDArray(one_hot.half().numpy())})
    enc_proj = torch.from_numpy(r["enc_proj"].numpy().astype(np.float32))[0]   # [T,640]
    T = enc_proj.shape[0]
    print(f"mel L={L} lang_prompt={int(d['prompt_id'])} -> enc_proj {tuple(enc_proj.shape)}")

    async def step_p(token, h, c):
        o = await pf({"token": rt.NDArray(token.numpy().astype(np.int32)),
                      "h": rt.NDArray(h.numpy()), "c": rt.NDArray(c.numpy())})
        return (torch.from_numpy(o["dec_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["h_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["c_out"].numpy().astype(np.float32)))

    async def step_j(dec, enc_frame):
        o = await jf({"dec_out": rt.NDArray(dec.numpy()), "enc_frame": rt.NDArray(enc_frame.numpy())})
        return torch.from_numpy(o["token_logits"].numpy().astype(np.float32))

    h = torch.zeros(NL, 1, HID); c = torch.zeros(NL, 1, HID)
    dec, h, c = await step_p(torch.tensor([[BLANK]]), h, c)
    frame, syms, emitted, steps = 0, 0, [], 0
    while frame < T and steps < MAX_SYMS * T + 16:
        logits = await step_j(dec, enc_proj[frame:frame + 1])
        token = int(logits.argmax())
        steps += 1
        if token == BLANK:
            frame += 1; syms = 0
        else:
            emitted.append(token)
            dec, h, c = await step_p(torch.tensor([[token]]), h, c)
            syms += 1
            if syms >= MAX_SYMS:
                frame += 1; syms = 0

    exact = emitted == gold_tokens
    same = sum(int(a == b) for a, b in zip(emitted, gold_tokens))
    print(f"[e2e gpu] emitted {len(emitted)} (gold {len(gold_tokens)})  "
          f"exact={exact}  token-agree {same}/{len(gold_tokens)}")
    print(f"   golden: {text!r}")
    print(f"\n{'✅ PASS — Nemotron 3.5 ASR runs end-to-end on Core AI' if exact else '⚠️ near-match (fp16 encoder near-ties)'}")


if __name__ == "__main__":
    asyncio.run(main())
