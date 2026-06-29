# Community port — NOT an Apple model.
"""S5 oracle gate: Core AI engine vs the torch ternary reference for BitCPM-8B.

The oracle is BitCPM8B on CPU (the TQ2_0-dequantized forward = exactly what the engine runs),
NOT HF (HF would load the bf16 latent master = a different model). Feeds identical raw token ids
to both, compares the greedy continuation token-for-token. Faithful export+MSL => token-exact.

  LLM_RUNNER=~/code/coreai/coreai-models/.build/out/Products/Release/llm-runner \
    .venv/bin/python ../coreai-models-community/conversion/bitcpm/gate.py <bundle_dir> [n]
"""
import asyncio
import sys

import numpy as np
import torch

import coreai.runtime as rt
from coreai_models.models.macos.bitcpm import BitCPMConfig, load_bitcpm8b_from_gguf
from coreai_models.primitives.macos.cache import KVCache
from transformers import AutoTokenizer

GGUF = "/Users/majimadaisuke/code/coreai/_bitcpm_ckpt/bitcpm4-8b-tq2_0.gguf"
HF = "/Users/majimadaisuke/code/coreai/_bitcpm_ckpt/hf"
DTYPE = torch.float16
CAP = 128

PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
    "Question: What is 17 + 25? Answer:",
]


@torch.no_grad()
def oracle_ids(model, cfg, ids, n):
    buf = ids.shape[1] + n + 8
    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = buf
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved
    out, cur, pos0 = [], ids.int(), 0
    for _ in range(n):
        T = cur.shape[1]
        position_ids = torch.arange(pos0 + T, dtype=torch.int32).unsqueeze(0)
        logits = model(cur, position_ids, k_cache, v_cache)
        nxt = int(logits[0, -1].float().argmax())
        out.append(nxt)
        pos0 += T
        cur = torch.tensor([[nxt]], dtype=torch.int32)
    return out


async def engine_ids(fn, prompt_ids, n):
    """S=1 decode loop on the engine (M=1 ternary-kernel contract); state = KV, mutated in place."""
    cfg = BitCPMConfig(); cfg.max_position_embeddings = CAP
    k0, v0 = KVCache.create_cache_tensors(cfg, dtype=torch.float16)
    key = rt.NDArray(np.ascontiguousarray(k0.numpy()))
    val = rt.NDArray(np.ascontiguousarray(v0.numpy()))

    async def step(token, pos):
        res = await asyncio.wait_for(fn(
            inputs={"input_ids": rt.NDArray(np.array([[token]], dtype=np.int32)),
                    "position_ids": rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None])},
            state={"keyCache": key, "valueCache": val}), timeout=120)
        return int(np.asarray(res["logits"].numpy())[0, -1].argmax())

    nxt = None
    for p, t in enumerate(prompt_ids):
        nxt = await step(t, p)
    out, pos = [], len(prompt_ids)
    for _ in range(n):
        out.append(nxt)
        nxt = await step(nxt, pos)
        pos += 1
    return out


async def main():
    aimodel = str(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    print("loading BitCPM-8B torch oracle (CPU)...", flush=True)
    model, _ = load_bitcpm8b_from_gguf(GGUF, dtype=DTYPE)
    cfg = model.config
    m = await rt.AIModel.load(
        aimodel, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")

    passes = 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        ref = oracle_ids(model, cfg, ids, n)
        eng = await engine_ids(fn, ids[0].tolist(), n)
        common = next((i for i, (a, b) in enumerate(zip(ref, eng)) if a != b), min(len(ref), len(eng)))
        exact = ref == eng
        passes += exact
        print(f"{'OK ' if exact else '~~ '}{p!r}  common {common}/{n}", flush=True)
        if not exact:
            print(f"     REF: {tok.decode(ref)!r}")
            print(f"     ENG: {tok.decode(eng)!r}")
    print(f"\n=== engine==torch token-exact greedy: {passes}/{len(PROMPTS)} ===")


if __name__ == "__main__":
    asyncio.run(main())
