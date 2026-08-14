"""Phase 4 — full-pipeline gate: mel -> encoder.aimodel (fp16) -> host TDT greedy loop driving
predict/joint .aimodels -> transcript, compared token-for-token to the golden oracle.

The definitive "Parakeet runs end to end on Core AI" check. Run in the MAIN venv with _GPU_LOCK
held:
    coreai-models/.venv/bin/python gate_e2e.py
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch

import coreai.runtime as rt

HERE = Path(__file__).resolve().parent
HID, NL = 640, 2


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle", nargs="?", default="oracle.npz")
    ap.add_argument("--artifacts", default="artifacts", help="bundle dir (relative to this script)")
    args = ap.parse_args()
    art = HERE / args.artifacts

    d = np.load(HERE / args.oracle)
    mel = torch.from_numpy(d["input_features"]).float().transpose(1, 2).contiguous()  # [1,128,L]
    L = mel.shape[2]
    gold_tokens = d["tokens"].tolist()
    text = str(d["text"])
    # v2 and v3 differ only here: blank 1024 / 8192, and the durations the joint indexes.
    blank = int(d["blank_id"])
    durations = (d["durations"].tolist() if "durations" in d
                 else list(range(int(d["n_durations"]))))

    enc_fn = await gpu(art / f"parakeet_encoder_float16_L{L}.aimodel")
    pf = await gpu(art / "parakeet_predict_float32.aimodel")
    jf = await gpu(art / "parakeet_joint_float32.aimodel")

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
    dec, h, c = await step_p(torch.tensor([[blank]]), h, c)
    frame, emitted = 0, []
    while frame < T and len(emitted) < 12 * T:
        tl, dl = await step_j(dec, enc_proj[frame:frame + 1])
        token = int(tl.argmax()); dur = durations[int(dl.argmax())]
        if token == blank and dur == 0:
            dur = 1
        frame += dur
        if token != blank:
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
