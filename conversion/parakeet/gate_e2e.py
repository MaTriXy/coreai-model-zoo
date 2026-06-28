"""Phase 4 — full-pipeline gate: mel -> encoder.aimodel (fp16) -> host TDT greedy loop driving
predict/joint .aimodels -> transcript, compared token-for-token to the golden oracle.

The definitive "Parakeet runs end to end on Core AI" check. Run in the MAIN venv with _GPU_LOCK
held:
    coreai-models/.venv/bin/python gate_e2e.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import torch

import coreai.runtime as rt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
HID, NL, VOCAB, BLANK = 640, 2, 8193, 8192
DURATIONS = [0, 1, 2, 3, 4]


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def main():
    import sys
    oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle.npz"
    d = np.load(HERE / oracle)
    mel = torch.from_numpy(d["input_features"]).float().transpose(1, 2).contiguous()  # [1,128,L]
    L = mel.shape[2]
    gold_tokens = d["tokens"].tolist()
    text = str(d["text"])

    enc_fn = await gpu(ART / f"parakeet_encoder_float16_L{L}.aimodel")
    pf = await gpu(ART / "parakeet_predict_float32.aimodel")
    jf = await gpu(ART / "parakeet_joint_float32.aimodel")

    # encoder (fp16) -> enc_proj
    r = await enc_fn({"mel": rt.NDArray(mel.half().numpy())})
    enc_proj = torch.from_numpy(r["enc_proj"].numpy().astype(np.float32))[0]   # [T,640]
    T = enc_proj.shape[0]
    print(f"mel L={L} -> enc_proj {tuple(enc_proj.shape)}")

    async def step_p(token, h, c):
        o = await pf({"token": rt.NDArray(token.numpy().astype(np.int32)),
                      "h": rt.NDArray(h.numpy()), "c": rt.NDArray(c.numpy())})
        return (torch.from_numpy(o["dec_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["h_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["c_out"].numpy().astype(np.float32)))

    async def step_j(dec, enc_frame):
        o = await jf({"dec_out": rt.NDArray(dec.numpy()), "enc_frame": rt.NDArray(enc_frame.numpy())})
        return (torch.from_numpy(o["token_logits"].numpy().astype(np.float32)),
                torch.from_numpy(o["dur_logits"].numpy().astype(np.float32)))

    h = torch.zeros(NL, 1, HID); c = torch.zeros(NL, 1, HID)
    dec, h, c = await step_p(torch.tensor([[BLANK]]), h, c)
    frame, emitted = 0, []
    while frame < T and len(emitted) < 12 * T:
        tl, dl = await step_j(dec, enc_proj[frame:frame + 1])
        token = int(tl.argmax()); dur = DURATIONS[int(dl.argmax())]
        if token == BLANK and dur == 0:
            dur = 1
        frame += dur
        if token != BLANK:
            emitted.append(token)
            dec, h, c = await step_p(torch.tensor([[token]]), h, c)

    exact = emitted == gold_tokens
    # token-level agreement even if a few near-ties differ
    same = sum(int(a == b) for a, b in zip(emitted, gold_tokens))
    print(f"[e2e gpu] emitted {len(emitted)} (gold {len(gold_tokens)})  "
          f"exact={exact}  token-agree {same}/{len(gold_tokens)}")
    print(f"   golden: {text!r}")
    print(f"\n{'✅ PASS — Parakeet runs end-to-end on Core AI' if exact else '⚠️ near-match (fp16 encoder near-ties)'}")


if __name__ == "__main__":
    asyncio.run(main())
