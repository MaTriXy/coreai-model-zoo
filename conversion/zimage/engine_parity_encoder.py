"""Parity of the exported Core AI text-encoder bundle vs the fp32 oracle cap.

Reproduces the pipeline host prep (chat-template tokenize -> right-pad to L ->
embed_tokens gather -> causal+padding 4D mask), runs the encoder graph, slices to
the valid length, and compares the penultimate hidden to the oracle cap embeds.

  python engine_parity_encoder.py <bundle.aimodel> --L 64 --dtype bf16
"""
import argparse
import asyncio
import json
import os

import numpy as np
import torch

import coreai.runtime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")


def build_encoder_inputs(tok, te, prompt, L, dtype):
    """-> inputs_embeds [1,L,2560], mask4d [1,1,L,L], valid length Lv."""
    s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids_valid = tok(s, return_tensors="pt").input_ids
    Lv = ids_valid.shape[1]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((1, L), pad_id, dtype=torch.long)
    ids[0, :Lv] = ids_valid[0, :Lv]
    with torch.no_grad():
        emb = te.embed_tokens(ids)
    neg = torch.finfo(dtype).min
    m = torch.triu(torch.full((L, L), neg), 1)
    m[:, Lv:] = neg                                   # mask padding keys
    return emb.to(dtype), m[None, None].to(dtype), Lv


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    meta = json.load(open(os.path.join(OUT, "meta.json")))
    Lc, prompt = meta["cap_cond_L"], meta["prompt"]
    cap = torch.from_numpy(np.fromfile(os.path.join(OUT, "cap_cond.f32"), "<f4")).reshape(1, Lc, 2560).float()[0]

    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    print("[enc-parity] loading tokenizer + embed_tokens ...", flush=True)
    M = "Tongyi-MAI/Z-Image-Turbo"
    tok = AutoTokenizer.from_pretrained(M, subfolder="tokenizer")
    te = Qwen3Model.from_pretrained(M, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
    emb, mask, Lv = build_encoder_inputs(tok, te, prompt, args.L, dtype)
    print(f"[enc-parity] valid tokens {Lv} (oracle {Lc})", flush=True)

    fn = (await rt.AIModel.load(args.bundle, rt.SpecializationOptions.default())).load_function("main")
    r = await fn(inputs={"inputs_embeds": rt.NDArray(emb.contiguous()),
                         "mask": rt.NDArray(mask.contiguous())})
    pen = torch.as_tensor(r["penultimate"].numpy().astype(np.float32))[0, :Lc]
    c = float(np.corrcoef(pen.flatten().numpy(), cap.flatten().numpy())[0, 1])
    print(f"[enc-parity] penultimate vs oracle cap: corr {c:.6f}  max|d| {float((pen-cap).abs().max()):.3e}  "
          f"{'PASS' if c > 0.999 else 'CHECK'}")


if __name__ == "__main__":
    asyncio.run(main())
