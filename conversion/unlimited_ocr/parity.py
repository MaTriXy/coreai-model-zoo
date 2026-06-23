"""Step (1): EAGER parity of the coreai-primitive decoder refactor vs the fp32 oracle.

Rebuilds the Unlimited-OCR R-SWA decoder on the Core AI macOS primitives
(`model.py`: SwitchGLU MoE / RMSNorm / SDPA composite / Llama RoPE),
loads the checkpoint, runs the SAME teacher-forced full-R-SWA-mask forward the
plain-torch parity used, and gates logits vs `out/_oracle/oracle_tensors.npz`.

Confirms the primitives reproduce the verified math before metalize + export.

Run (coreai venv has coreai_torch + coreai_models):
    coreai-models/.venv/bin/python parity.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from model import ocr_decoder_from_safetensors

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
REF = os.path.join(HERE, "out/_oracle/oracle_tensors.npz")


def main() -> None:
    torch.manual_seed(0)
    model = ocr_decoder_from_safetensors(CKPT, target_dtype=torch.float32).eval()

    z = np.load(REF)
    dec = torch.from_numpy(z["dec_embeds"]).float()  # [1,Lm,1280]
    ids = torch.from_numpy(z["token_ids"]).long()  # [1,501]
    Lm = dec.shape[1]
    Ttot = ids.shape[1]
    with torch.no_grad():
        gen_emb = model.model.embed_tokens(ids[:, Lm:Ttot])  # embed generated tokens
        embeds = torch.cat([dec, gen_emb], dim=1)  # [1,Ttot,1280]
        pos = torch.arange(Ttot)[None]
        h = model.model.forward(embeds, pos, Lm)  # full R-SWA mask, no cache
        logits = model.lm_head(h)  # [1,Ttot,VOCAB]

    print(f"[parity] Lm={Lm} T={Ttot}  logits {tuple(logits.shape)}", flush=True)
    steps = sorted(int(k[len("logits_step"):]) for k in z.files if k.startswith("logits_step"))
    allcos, allflip = [], 0
    for s in steps:
        ref = torch.from_numpy(z[f"logits_step{s}"]).float()[0]  # [VOCAB]
        mine = logits[0, Lm - 1 + s].float()  # step s predicts token Lm+s
        cos = F.cosine_similarity(mine, ref, dim=0).item()
        flip = int(mine.argmax() != ref.argmax())
        allflip += flip
        allcos.append(cos)
        print(f"  step{s:4d}: cos={cos:.6f}  argmax {'MATCH' if not flip else 'FLIP'} "
              f"(mine={int(mine.argmax())} ref={int(ref.argmax())})", flush=True)
    print(f"\nVERDICT: cos min={min(allcos):.6f} mean={np.mean(allcos):.6f}  flips={allflip}/{len(steps)}")
    print("✅ PASS" if min(allcos) > 0.999 and allflip == 0 else "❌ FAIL")


if __name__ == "__main__":
    main()
