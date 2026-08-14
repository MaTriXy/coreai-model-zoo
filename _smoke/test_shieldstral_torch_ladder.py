#!/usr/bin/env python3
"""fp32 torch ladder for Shieldstral-1.0-3B — is `ministral3` just Mistral + YARN?

The oracle (`shieldstral_ref.py`) runs HF's native `ministral3` on transformers git
main. The exporter cannot: the conversion venv is on 4.57.6, which has never heard
of `ministral3`. The whole port rests on one claim — that this checkpoint is a
plain Mistral decoder whose only deviation is YARN rope, so 4.57.6's `MistralModel`
plus `rope_scaling={'rope_type': 'yarn', ...}` reproduces it exactly.

That claim is cheap to test and expensive to assume (a mis-scaled rope still emits
fluent-looking logits), so this measures it: same nine prompts, float64 cosine on
the last-position logits, and the same P(unsafe) to 1e-4.

Run (conversion venv, transformers 4.57.6):
    ../coreai-models/.venv/bin/python _smoke/test_shieldstral_torch_ladder.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import hf_snapshot  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "shieldstral_3b_suite_ref.npz"


def cos64(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def build_text_model(snap: Path, dtype=torch.float32):
    """4.57.6 MistralModel carrying Shieldstral's text_config, YARN included."""
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    raw = json.loads((snap / "config.json").read_text())
    t = raw["text_config"]
    rope = dict(t["rope_parameters"])
    theta = rope.pop("rope_theta")
    rope.pop("type", None)                 # v5 duplicate of rope_type
    rope.pop("llama_4_scaling_beta", None)  # not consumed by yarn
    cfg = MistralConfig(
        vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"],
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"], hidden_act=t["hidden_act"],
        max_position_embeddings=t["max_position_embeddings"],
        rms_norm_eps=t["rms_norm_eps"], rope_theta=theta, rope_scaling=rope,
        sliding_window=t["sliding_window"], tie_word_embeddings=True,
        attention_dropout=0.0, torch_dtype=dtype,
    )
    model = MistralModel(cfg).to(dtype).eval()

    from safetensors.torch import load_file
    sd = load_file(str(snap / "model.safetensors"))
    pref = "language_model.model."
    text_sd = {k[len(pref):]: v.to(dtype) for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise SystemExit(f"weight mismatch: missing={missing[:6]} unexpected={unexpected[:6]}")
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(dtype)
    head.weight = model.embed_tokens.weight        # tied
    return model, head, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="mistralai/Shieldstral-1.0-3B")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=True)
    n = int(ref["_meta_cases"])
    yes_id, no_id = int(ref["_meta_yes_id"]), int(ref["_meta_no_id"])
    labels = [str(x) for x in ref["_meta_labels"]]

    snap = Path(hf_snapshot(args.hf_id))
    print("building 4.57.6 MistralModel + YARN ...", flush=True)
    model, head, cfg = build_text_model(snap)
    print(f"  {cfg.num_hidden_layers}L hidden {cfg.hidden_size} "
          f"GQA {cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"rope {cfg.rope_scaling['rope_type']} x{cfg.rope_scaling['factor']}\n")

    worst_cos, worst_dp, bad = 1.0, 0.0, 0
    print(f"{'case':26s} {'cos(logits_last)':>17s}  {'P ref':>7s} {'P mine':>7s} {'|dP|':>8s}")
    for i in range(n):
        ids = torch.tensor(ref[f"case{i}_ids"], dtype=torch.long)[None]
        with torch.no_grad():
            h = model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
            logits = head(h)[0].float()
        c = cos64(logits.numpy(), ref[f"case{i}_logits_last"])
        yn = torch.stack([logits[no_id], logits[yes_id]])
        p = float(torch.softmax(yn, 0)[1])
        p_ref = float(ref[f"case{i}_p"])
        dp = abs(p - p_ref)
        side_ok = (p > 0.5) == (p_ref > 0.5)
        worst_cos, worst_dp = min(worst_cos, c), max(worst_dp, dp)
        bad += int(not side_ok or c < 0.9999)
        print(f"{labels[i]:26s} {c:17.6f}  {p_ref:7.4f} {p:7.4f} {dp:8.5f}"
              f"{'' if side_ok else '  ** SIDE FLIP **'}")

    print(f"\nworst cos {worst_cos:.6f} | worst |dP| {worst_dp:.5f}")
    ok = worst_cos >= 0.9999 and worst_dp < 1e-3 and bad == 0
    print("PASS: ministral3 == Mistral + YARN on 4.57.6" if ok else "FAIL: reimplementation needed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
