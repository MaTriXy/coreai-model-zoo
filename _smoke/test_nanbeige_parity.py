"""CPU fp32 parity smoke for the Nanbeige4.1-3B plain-Llama overlay.

Validates that the `models/macos/llama.py` port reproduces the HuggingFace
`LlamaForCausalLM` oracle on a teacher-forced prompt (top-1 per position +
cosine), BEFORE any Core AI export. model_type is "llama" => the oracle is
native transformers (no trust_remote_code). Pure CPU / fp32 => non-colliding
with GPU work (diffgemma / coder-next exports).

`USE_HF_IMPL=true` makes the RoPE/SDPA composites use their HF-matching torch
reference (CPU-friendly, and isolates architecture correctness from the
composite-op kernel numerics, which the engine gate checks later).

Run:  cd ~/code/coreai/coreai-models && \
      USE_HF_IMPL=true DISABLE_BFLOAT16_CAST_FOR_LOGITS=true \
      .venv/bin/python ../coreai-models-community/_smoke/test_nanbeige_parity.py
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_HF_IMPL", "true")
os.environ.setdefault("DISABLE_BFLOAT16_CAST_FOR_LOGITS", "true")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ID = "Nanbeige/Nanbeige4.1-3B"
PROMPT = "The capital of France is"
DTYPE = torch.float32
CACHE_LEN = 512  # tiny KV cache for the smoke (clamp the 262K config)


def main() -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(HF_ID)
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(torch.int32)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.int32).unsqueeze(0)
    print(f"prompt={PROMPT!r}  tokens={seq}  ids={input_ids.tolist()[0]}", flush=True)

    # --- HF oracle (native llama, fp32) ---
    print("loading HF oracle (fp32) ...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=DTYPE).eval()
    with torch.no_grad():
        ref_logits = hf(input_ids.to(torch.long)).logits.float()  # [1, S, V]
    ref_top1 = ref_logits.argmax(-1)[0]
    print(f"oracle next-token argmax (last pos) = {tok.decode(ref_top1[-1])!r}", flush=True)
    del hf

    # --- Core AI port (plain-Llama overlay, fp32) ---
    print("loading Core AI port LlamaForCausalLM (fp32) ...", flush=True)
    from coreai_models.models.macos.llama import LlamaForCausalLM
    from coreai_models.primitives.macos.cache import KVCache

    model = LlamaForCausalLM.from_hf(HF_ID, max_context_length=CACHE_LEN, target_dtype=DTYPE).eval()
    cfg = model.config
    cfg.max_position_embeddings = CACHE_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    with torch.no_grad():
        port_logits = model(input_ids, position_ids, k_cache, v_cache).float()  # [1, S, V]
    port_top1 = port_logits.argmax(-1)[0]

    # --- compare ---
    cos = torch.nn.functional.cosine_similarity(
        ref_logits[0].double(), port_logits[0].double(), dim=-1
    )
    max_abs = (ref_logits - port_logits).abs().max().item()
    agree = int((ref_top1 == port_top1).sum())
    print(f"\n=== parity (S={seq}) ===", flush=True)
    print(f"top-1 agreement : {agree}/{seq}", flush=True)
    print(f"cosine/pos      : min={cos.min():.6f} mean={cos.mean():.6f}", flush=True)
    print(f"max abs logit Δ : {max_abs:.4f}", flush=True)
    print(f"port next-token : {tok.decode(port_top1[-1])!r}", flush=True)

    ok = agree == seq and cos.min() > 0.999
    print("\nPARITY PASS ✅" if ok else "\nPARITY FAIL ❌", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
