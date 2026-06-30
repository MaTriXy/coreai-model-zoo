# Community port — NOT an Apple model.
"""Torch parity gate for the LLaDA-8B backbone overlay (`llada.LLaDABackbone`) vs the OFFICIAL
`LLaDAModelLM` (d3LLM/d3LLM_LLaDA == GSAI-ML/LLaDA-8B-Instruct, modeling_llada.py).

Loads the real checkpoint into BOTH, feeds identical `input_ids` (a chunk of positions set to the
[MASK] token 126336, as in masked-diffusion inference), and checks cos of the per-layer hidden
states AND the final logits. Bidirectional, full forward, no KV cache.

  coreai-models/.venv/bin/python gate_llada_torch.py            # few-layer (fast) then full 32L
  coreai-models/.venv/bin/python gate_llada_torch.py --layers 4 # few-layer only
  coreai-models/.venv/bin/python gate_llada_torch.py --layers 32

Pass = min cos >= 0.999 across every hidden + logits.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "d3LLM_LLaDA")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref"))  # import official as the `llada_ref` package (relative imports)

from llada import LLaDABackbone, LLaDACfg, load_backbone  # noqa: E402

DTYPE = torch.float32
MASK_TOKEN = 126336
VOCAB = 126464
BUF = 64           # rope table length; gate uses T <= BUF
T = 32             # sequence length for the gate


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def load_weights(num_layers: int) -> dict:
    """Pull only the tensors needed for an `num_layers`-deep model, in fp32, via safe_open."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(WEIGHTS, "model.safetensors.index.json")))["weight_map"]
    want = {"model.transformer.wte.weight", "model.transformer.ln_f.weight",
            "model.transformer.ff_out.weight"}
    for li in range(num_layers):
        for p in ("attn_norm", "ff_norm", "q_proj", "k_proj", "v_proj", "attn_out",
                  "ff_proj", "up_proj", "ff_out"):
            want.add(f"model.transformer.blocks.{li}.{p}.weight")

    # group wanted keys by shard so each file is opened once
    by_shard: dict[str, list[str]] = {}
    for k in want:
        by_shard.setdefault(index[k], []).append(k)

    sd = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(WEIGHTS, shard), framework="pt") as f:
            for k in keys:
                sd[k] = f.get_tensor(k).to(DTYPE)
    return sd


def build_official(num_layers: int, sd: dict):
    from llada_ref.configuration_llada import LLaDAConfig
    from llada_ref.modeling_llada import LLaDAModelLM

    cfg_dict = json.load(open(os.path.join(WEIGHTS, "config.json")))
    cfg_dict["n_layers"] = num_layers
    cfg = LLaDAConfig(**cfg_dict)
    model = LLaDAModelLM(cfg, init_params=False).to(DTYPE).eval()
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.endswith(("cos_table", "sin_table"))
            and "rope" not in m and "cache" not in m]
    if miss:
        raise RuntimeError(f"official: {len(miss)} unloaded, e.g. {miss[:4]}")
    return model


def gate(num_layers: int) -> float:
    print(f"\n=== LLaDA backbone gate: {num_layers}L (T={T}, fp32) ===")
    sd = load_weights(num_layers)
    official = build_official(num_layers, sd)
    mine = load_backbone(sd, num_layers, BUF, DTYPE)
    del sd

    # identical input: random token ids with a contiguous [MASK] span (masked-diffusion canvas)
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, T), dtype=torch.long)
    ids[:, T // 2:] = MASK_TOKEN  # back half is the masked region to be denoised

    with torch.inference_mode():
        off = official(input_ids=ids, output_hidden_states=True)
        off_logits = off.logits                      # [1,T,vocab]
        off_hidden = off.hidden_states               # tuple len num_layers+1
        my_logits, my_hidden = mine(ids, return_hidden=True)

    worst = 1.0
    for i, (a, b) in enumerate(zip(my_hidden, off_hidden)):
        c = cos(a, b)
        worst = min(worst, c)
        tag = "embed/in" if i < num_layers else "post-ln_f"
        if i < 3 or i >= num_layers - 1:
            print(f"  hidden[{i:2d}] ({tag:9s}) cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}")

    lc = cos(my_logits, off_logits)
    worst = min(worst, lc)
    print(f"  logits            cos={lc:.6f}  {'OK' if lc >= 0.999 else 'FAIL'}")

    # token-level agreement on the masked region (sanity for the denoising loop later)
    my_pred = my_logits[0, T // 2:].argmax(-1)
    off_pred = off_logits[0, T // 2:].argmax(-1)
    agree = (my_pred == off_pred).float().mean().item()
    print(f"  masked argmax agreement = {agree*100:.1f}%")

    print(f"  >>> {num_layers}L min cos = {worst:.6f}  {'PASS' if worst >= 0.999 else 'FAIL'}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None,
                    help="layer count for a single gate; default runs 4L then 32L")
    args = ap.parse_args()

    runs = [args.layers] if args.layers else [4, 32]
    worst = 1.0
    for n in runs:
        worst = min(worst, gate(n))
    print(f"\n>>> overall min cos = {worst:.6f}  ->  {'GATE PASS' if worst >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if worst >= 0.999 else 1)


if __name__ == "__main__":
    main()
