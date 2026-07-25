# Community port — NOT an Apple model.
"""CPU parity: the EXPORT LLM module (`bitvla_llm.BitVLALLM`, inputs_embeds + KVCache decode loop,
ternary BitLinearMetal via torch_defn) reproduces the `bitvla_ref` action tokens. Also the
reference host-side driver for the engine gate (build embeds: text + 256 vision + text; prefill
one position at a time; greedy-generate 7 action tokens). No GPU.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/gate_llm_cpu.py [--num-layers 30]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitvla_ref as B  # noqa: E402

from coreai_models.models.macos.bitvla_llm import BitVLALLMConfig, load_bitvla_llm  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

CK = str(work_path("_bitvla_ckpt", "bitvla_bf16", "model.safetensors"))
ORACLE = str(work_path("_bitvla_ckpt", "oracle.npz"))


def build_inputs_embeds(tok, embed_w, img_embeds, instruction, dtype):
    """Same prompt as bitvla_ref: '<sys>Human: <image>\\n<q><|eot_id|>Assistant: ' (256 img embeds)."""
    pre = B.SYS + "Human: "
    post = "\n" + f"What action should the robot take to {instruction}?" + "<|eot_id|>Assistant: "
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=True).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    e_pre = F.embedding(pre_ids, embed_w)
    e_post = F.embedding(post_ids, embed_w)
    return torch.cat([e_pre, img_embeds.to(dtype), e_post], dim=1)


@torch.no_grad()
def decode_generate(model, cfg, embeds, embed_w, new=8, stop_ids=(128001,), dtype=torch.float16):
    """Prefill position-by-position (M=1) then greedy-generate. Mirrors the engine S=1 loop."""
    S = embeds.shape[1]
    buf = S + new + 4
    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = buf
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=dtype)
    cfg.max_position_embeddings = saved

    logits_last = None
    for t in range(S):                                   # prefill one token at a time
        emb_t = embeds[:, t:t + 1, :]
        position_ids = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
        logits_last = model(emb_t, position_ids, k_cache, v_cache)
    out = []
    pos = S
    for _ in range(new):
        nxt = int(logits_last[0, -1].float().argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        emb_t = F.embedding(torch.tensor([[nxt]]), embed_w).to(dtype)
        position_ids = torch.arange(pos + 1, dtype=torch.int32).unsqueeze(0)
        logits_last = model(emb_t, position_ids, k_cache, v_cache)
        pos += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=30)
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"])
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(work_path("_bitvla_ckpt", "bitvla_bf16")))

    z = np.load(ORACLE, allow_pickle=True)
    img_embeds = torch.from_numpy(z["img_embeds"]).float()      # use the validated projected embeds
    instr = str(z["instruction"])
    off_ids = [int(t) for t in z["action_ids"].tolist() if t != 128001][:7]

    print(f"loading export LLM ({args.num_layers}L, {args.dtype}) ...", flush=True)
    model, kernel, embed_w = load_bitvla_llm(CK, num_layers=args.num_layers, dtype=dtype)
    embeds = build_inputs_embeds(tok, embed_w, img_embeds, instr, dtype)
    print(f"inputs_embeds {tuple(embeds.shape)} (text + 256 image + text); generating ...", flush=True)
    mine = decode_generate(model, model.config, embeds, embed_w, new=8, dtype=dtype)

    # reference (full-recompute) action tokens on the same img_embeds
    llm_ref = B.BitNetLLM().float().eval(); B.load_llm(llm_ref)
    e_ref = B.action_prompt_embeds(tok, llm_ref, img_embeds, instr)
    ref_ids = B.generate(llm_ref, e_ref, new=8, stop_ids=(128001,))[:7]

    print("official ids:", off_ids)
    print("ref      ids:", ref_ids)
    print("export   ids:", mine)
    n_ref = sum(a == b for a, b in zip(mine, ref_ids))
    n_off = sum(a == b for a, b in zip(mine, off_ids))
    print(f"export vs ref: {n_ref}/{len(ref_ids)}   export vs official: {n_off}/{len(off_ids)}")
    print("PASS" if mine == ref_ids else "DIFF (check fp16-scale boundary flips)")


if __name__ == "__main__":
    main()
