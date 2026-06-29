# Community port — NOT an Apple model.
"""S5 engine gate: drive the BitCPM-8B static-ids .aimodel on the Core AI GPU engine in an S=1
decode loop (the M=1 ternary-kernel contract) and print the greedy continuation. Coherent output
('Paris') confirms the ternary kernel + mup + LongRoPE numerics are right ON THE ENGINE.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitcpm/engine_gen.py \
    exports/bitcpm_8b_decode_ternary_s1/bitcpm_8b_decode_ternary_s1.aimodel "The capital of France is" 8
"""
import asyncio
import sys

import numpy as np
import torch
import coreai.runtime as rt
from transformers import AutoTokenizer

from coreai_models.models.macos.bitcpm import BitCPMConfig
from coreai_models.primitives.macos.cache import KVCache

HF = "/Users/majimadaisuke/code/coreai/_bitcpm_ckpt/hf"
CAP = 128  # KV capacity (prompt + gen)


async def main():
    aimodel = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    cfg = BitCPMConfig()
    cfg.max_position_embeddings = CAP
    k0, v0 = KVCache.create_cache_tensors(cfg, dtype=torch.float16)
    print("KV state shape:", tuple(k0.shape), flush=True)
    key = rt.NDArray(np.ascontiguousarray(k0.numpy()))
    val = rt.NDArray(np.ascontiguousarray(v0.numpy()))

    m = await rt.AIModel.load(
        aimodel, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")

    async def step(token, pos):
        iid = rt.NDArray(np.array([[token]], dtype=np.int32))
        pid = rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None])
        res = await asyncio.wait_for(
            fn(inputs={"input_ids": iid, "position_ids": pid},
               state={"keyCache": key, "valueCache": val}), timeout=120)
        logits = res["logits"].numpy()
        return int(np.asarray(logits)[0, -1].argmax())

    # prefill the prompt as S=1 steps (chunkThreshold=1 contract); last step yields token 0
    nxt = None
    for p, t in enumerate(ids):
        nxt = await step(t, p)
    out = []
    pos = len(ids)
    for _ in range(n):
        out.append(nxt)
        print("  ->", nxt, repr(tok.decode([nxt])), flush=True)
        nxt = await step(nxt, pos)
        pos += 1
    print("ENGINE GENERATION:", repr(tok.decode(out)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
