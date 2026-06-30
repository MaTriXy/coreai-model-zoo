# Community port — NOT an Apple model.
"""S3+S4 engine gate: drive the exported BitVLA bundle end-to-end on the Core AI GPU engine and
compare the predicted 7-DoF action to the torch reference (and the official oracle).

Pipeline: pixel_values -> [vision .aimodel] -> img_embeds[1,256,2560] -> host builds inputs_embeds
(text-pre + 256 image + text-post via the embed_tokens table) -> [LLM .aimodel] S=1 decode loop
(prefill one position at a time, then greedy 7 action tokens; action_head -> argmax over 256 ->
token_id = ACT_LO + j) -> detokenize (256-bin + BOUNDS_Q99). Confirms the ternary kernel + splice
+ detok are right ON THE ENGINE.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/engine_gate.py \
    --vision exports/bitvla_vision_int8lin/bitvla_vision_int8lin.aimodel \
    --llm exports/bitvla_llm_decode_ternary_s1/bitvla_llm_decode_ternary_s1.aimodel
GPU: grab _GPU_LOCK.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import coreai.runtime as rt
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/majimadaisuke/code/coreai/coreai-models-community/conversion/bitvla")
import bitvla_ref as B  # noqa: E402

from coreai_models.models.macos.bitvla_llm import BitVLALLMConfig, ACT_LO, N_ACTION_BINS  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

CKDIR = "/Users/majimadaisuke/code/coreai/_bitvla_ckpt/bitvla_bf16"
CK = f"{CKDIR}/model.safetensors"
ORACLE = "/Users/majimadaisuke/code/coreai/_bitvla_ckpt/oracle.npz"
CAP = 320


def gpu():
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())


async def run(args):
    tok = AutoTokenizer.from_pretrained(CKDIR)
    z = np.load(ORACLE, allow_pickle=True)
    pv = z["pixel_values"].astype(np.float16)
    instr = str(z["instruction"])
    off7 = [int(t) for t in z["action_ids"].tolist() if t != 128001][:7]
    embed_w = load_file(CK)["language_model.model.embed_tokens.weight"].float()

    # 1) vision .aimodel -> img_embeds
    vm = await rt.AIModel.load(args.vision, gpu())
    vfn = vm.load_function("main")
    vres = await asyncio.wait_for(
        vfn(inputs={"pixel_values": rt.NDArray(np.ascontiguousarray(pv))}), timeout=300)
    img = torch.from_numpy(vres["img_embeds"].numpy().astype(np.float32)).reshape(1, 256, 2560)
    print(f"[vision] img_embeds {tuple(img.shape)} std {img.std():.3f}", flush=True)

    # 2) host: build inputs_embeds (text-pre + 256 image + text-post)
    pre = B.SYS + "Human: "
    post = "\n" + f"What action should the robot take to {instr}?" + "<|eot_id|>Assistant: "
    e_pre = F.embedding(tok(pre, return_tensors="pt", add_special_tokens=True).input_ids, embed_w)
    e_post = F.embedding(tok(post, return_tensors="pt", add_special_tokens=False).input_ids, embed_w)
    embeds = torch.cat([e_pre, img, e_post], dim=1).to(torch.float16)
    S = embeds.shape[1]
    print(f"[host] inputs_embeds {tuple(embeds.shape)}", flush=True)

    # 3) LLM .aimodel S=1 decode loop
    cfg = BitVLALLMConfig(max_position_embeddings=CAP, action_head=True)
    k0, v0 = KVCache.create_cache_tensors(cfg, dtype=torch.float16)
    key = rt.NDArray(np.ascontiguousarray(k0.numpy()))
    val = rt.NDArray(np.ascontiguousarray(v0.numpy()))
    lm = await rt.AIModel.load(args.llm, gpu())
    lfn = lm.load_function("main")

    async def step(emb_1x1xH, pos):
        ie = rt.NDArray(np.ascontiguousarray(emb_1x1xH.numpy().astype(np.float16)))
        pid = rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None])
        res = await asyncio.wait_for(
            lfn(inputs={"inputs_embeds": ie, "position_ids": pid},
                state={"keyCache": key, "valueCache": val}), timeout=120)
        return np.asarray(res["logits"].numpy())[0, -1]

    def to_token(logits):
        # action_head export -> 256 rows (token = ACT_LO + j); full head -> argmax over vocab
        return ACT_LO + int(logits.argmax()) if logits.shape[-1] == N_ACTION_BINS else int(logits.argmax())

    logits = None
    for t in range(S):
        logits = await step(embeds[:, t:t + 1, :], t)
    out = []
    pos = S
    for _ in range(7):
        tid = to_token(logits)
        out.append(tid)
        emb = F.embedding(torch.tensor([[tid]]), embed_w).to(torch.float16)
        logits = await step(emb, pos)
        pos += 1

    ns = json.load(open(f"{CKDIR}/config.json"))["norm_stats"]
    key_ds = args.unnorm_key
    print(f"[engine] action ids: {out}", flush=True)
    print(f"[oracle] action ids: {off7}", flush=True)
    print(f"  match vs official: {sum(a==b for a,b in zip(out,off7))}/7", flush=True)
    print(f"[engine] 7-DoF ({key_ds}): {np.round(B.detokenize_action(out, ns, key_ds),4)}", flush=True)
    print(f"[oracle] 7-DoF ({key_ds}): {np.round(B.detokenize_action(off7, ns, key_ds),4)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--unnorm-key", default="bridge_orig")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
