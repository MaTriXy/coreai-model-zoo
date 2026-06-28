# Community port — NOT an Apple model.
"""Torch parity gate for the VoxCPM2 (2B) MiniCPM4 backbone: my fixed-shape overlay
(`minicpm4.MiniCPM4Backbone`, prefill-via-decode recipe) vs the OFFICIAL `MiniCPMModel`
(github OpenBMB/VoxCPM, pulled to _ref_v2/official_minicpm4/). Loads the real VoxCPM2 weights
into both, feeds identical seeded `inputs_embeds`, and checks cos of every hidden.

base_lm  = 28L, embed_tokens, rope (short_factor).   residual_lm = 8L, vocab 0, NO rope.
Pass = cos >= 0.999 on both prefill (batched) and decode (per-step) for base AND residual.

  coreai-models/.venv/bin/python gate_v2_backbone_torch.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref_v2"))

from minicpm4 import MiniCPM4Backbone, MiniCPM4Cfg, build_kv_state  # noqa: E402
from official_minicpm4 import MiniCPM4Config, MiniCPMModel  # noqa: E402

DTYPE = torch.float32
BUF = 64
T = 16  # prefill length for the gate


def snap() -> str:
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def load_sd() -> dict:
    from safetensors.torch import load_file
    return load_file(snap() + "/model.safetensors")


def vox2_lm_config() -> dict:
    return json.load(open(snap() + "/config.json"))["lm_config"]


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def official(lm_cfg: dict, n_layers: int, vocab: int, no_rope: bool) -> MiniCPMModel:
    c = dict(lm_cfg)
    c["num_hidden_layers"] = n_layers
    c["vocab_size"] = vocab
    c["no_rope"] = no_rope
    m = MiniCPMModel(MiniCPM4Config(**c)).to(DTYPE).eval()
    return m


def mine(sd: dict, prefix: str, lm_cfg: dict, n_layers: int, vocab: int, no_rope: bool) -> MiniCPM4Backbone:
    from minicpm4 import load_backbone
    return load_backbone(
        sd, prefix, n_layers, vocab, BUF, DTYPE,
        hidden_size=lm_cfg["hidden_size"], intermediate_size=lm_cfg["intermediate_size"],
        num_attention_heads=lm_cfg["num_attention_heads"],
        num_key_value_heads=lm_cfg["num_key_value_heads"],
        head_dim=lm_cfg["kv_channels"], short_factor=lm_cfg["rope_scaling"]["short_factor"],
        no_rope=no_rope,
    )


def gate_one(sd, lm_cfg, name, prefix, n_layers, vocab, no_rope) -> list:
    print(f"\n=== {name} ({n_layers}L, vocab={vocab}, no_rope={no_rope}) ===")
    off = official(lm_cfg, n_layers, vocab, no_rope)
    # load official from the same checkpoint
    sub = {k[len(prefix):]: v.to(DTYPE) for k, v in sd.items() if k.startswith(prefix)}
    miss, unexp = off.load_state_dict(sub, strict=False)
    miss = [m for m in miss if "inv_freq" not in m and "cached" not in m]
    if miss:
        raise RuntimeError(f"official {name}: unloaded {miss[:4]}")
    mybb = mine(sd, prefix, lm_cfg, n_layers, vocab, no_rope)

    torch.manual_seed(0)
    H = lm_cfg["hidden_size"]
    emb = torch.randn(1, T, H, dtype=DTYPE)

    results = []
    # ---- prefill (batched) ----
    with torch.inference_mode():
        off_h, _ = off(inputs_embeds=emb, is_causal=True)        # [1,T,H]
        k_cache, v_cache = build_kv_state(mybb.cfg, BUF, DTYPE)
        my_h = mybb.prefill(emb, k_cache, v_cache)               # [1,T,H]
    c = cos(my_h, off_h)
    print(f"  prefill q={T}  cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}")
    results.append(c)

    # ---- decode (per-step), fresh caches on both sides ----
    off.setup_cache(1, BUF, torch.device("cpu"), DTYPE)
    k2, v2 = build_kv_state(mybb.cfg, BUF, DTYPE)
    with torch.inference_mode():
        for i in range(T):
            step = emb[:, i, :]                                   # [1,H]
            off_s = off.forward_step(step, torch.tensor([i]))     # [1,H]
            my_s = mybb.decode(step.reshape(1, 1, H), torch.tensor([i], dtype=torch.int32), k2, v2)
            c = cos(my_s, off_s)
            results.append(c)
            if i < 3 or i == T - 1:
                print(f"  decode[{i:2d}] cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}")
    return results


def main():
    sd = load_sd()
    lm = vox2_lm_config()
    print(f"[cfg] hidden={lm['hidden_size']} heads={lm['num_attention_heads']} "
          f"kv={lm['num_key_value_heads']} kv_channels={lm['kv_channels']} "
          f"inter={lm['intermediate_size']} use_mup={lm.get('use_mup')}")
    allc = []
    allc += gate_one(sd, lm, "base_lm", "base_lm.", 28, 73448, no_rope=False)
    allc += gate_one(sd, lm, "residual_lm", "residual_lm.", 8, 0, no_rope=True)
    lo = min(allc)
    print(f"\n>>> min cos = {lo:.6f}  ->  {'GATE PASS' if lo >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    main()
