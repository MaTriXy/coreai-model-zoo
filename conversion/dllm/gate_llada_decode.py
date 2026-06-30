# Community port — NOT an Apple model.
"""Decode-loop parity gate: my host loop (`generate_llada.generate`) on my overlay
(`llada.LLaDABackbone`) vs the OFFICIAL d3LLM `generate` on the official `LLaDAModelLM`.

Both are no-cache full-forward, temperature 0 (deterministic). The reference `generate` and
`get_transfer_index_entropy` below are copied VERBATIM from
`_ref/d3LLM/d3llm/d3llm_LLaDA/d3llm_llada_generate_util.py` (so the gate doesn't depend on importing
that module's heavy model deps). Pass = generated token ids identical.

  coreai-models/.venv/bin/python gate_llada_decode.py [--gen 32 --block 32 --layers 32]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "d3LLM_LLaDA")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref"))

from llada import load_backbone                                    # noqa: E402
from generate_llada import generate as my_generate                # noqa: E402
from gate_llada_torch import load_weights, build_official, DTYPE  # noqa: E402

MASK_ID = 126336
EOS_ID = 126081


# --------------------------------------------------------------------------- #
# VERBATIM reference (d3llm_llada_generate_util.py) — oracle loop
# --------------------------------------------------------------------------- #
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index_entropy(logits, temperature, remasking, mask_index, x,
                               num_transfer_tokens, entropy_threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    if remasking == "low_confidence":
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
    elif remasking == "random":
        entropy = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if entropy_threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(entropy_for_selection.shape[0]):
        _, select_index = torch.topk(entropy_for_selection[j], k=num_transfer_tokens[j], largest=False)
        transfer_index[j, select_index] = True
        if entropy_threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if entropy[j, select_index[k]] > entropy_threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


@torch.no_grad()
def ref_generate(model, prompt, gen_length=32, block_length=32, temperature=0.0,
                 remasking="low_confidence", mask_id=126336, threshold=0.5):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    num_blocks = gen_length // block_length
    nfe = 0
    for num_block in range(num_blocks):
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            x0, transfer_index = get_transfer_index_entropy(
                logits, temperature, remasking, mask_index, x, None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length:
                   prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--prompt", type=str,
                    default="What is 12 multiplied by 8? Answer with just the number.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long).unsqueeze(0)
    L = ids.shape[1]
    buf = L + args.gen
    print(f"[decode-gate] prompt_len={L} gen={args.gen} block={args.block} layers={args.layers}")

    sd = load_weights(args.layers)
    official = build_official(args.layers, sd)
    mine = load_backbone(sd, args.layers, buf, DTYPE)
    del sd

    print("running official reference loop ...")
    ref_x, ref_nfe = ref_generate(official, ids, gen_length=args.gen, block_length=args.block,
                                  threshold=0.5, mask_id=MASK_ID)
    print("running my loop on my overlay ...")
    my_x, my_nfe = my_generate(lambda z: mine(z), ids, gen_length=args.gen, block_length=args.block,
                               threshold=0.5, mask_id=MASK_ID)

    ref_gen = ref_x[0, L:]
    my_gen = my_x[0, L:]
    same = bool((ref_gen == my_gen).all())
    n_match = int((ref_gen == my_gen).sum())
    print(f"\n  ref  nfe={ref_nfe}  tokens={ref_gen.tolist()}")
    print(f"  mine nfe={my_nfe}  tokens={my_gen.tolist()}")
    print(f"  token match: {n_match}/{args.gen}  nfe match: {ref_nfe == my_nfe}")
    print(f"\n  ref  text: {tok.decode(ref_gen, skip_special_tokens=True)!r}")
    print(f"  mine text: {tok.decode(my_gen, skip_special_tokens=True)!r}")
    print(f"\n>>> DECODE GATE {'PASS' if same and ref_nfe == my_nfe else 'FAIL'}")
    sys.exit(0 if same and ref_nfe == my_nfe else 1)


if __name__ == "__main__":
    main()
