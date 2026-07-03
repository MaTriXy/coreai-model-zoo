"""Eager numerics gate for Ornith-1.0-9B (qwen3_5 dense hybrid) — the 27B method.

Teacher-forced single-step (S=1, loop-free, the shipped decode semantics) argmax
vs the fp32 HF oracle (_smoke/ornith9b_ref.pt), under the margin rule: a flip at
an oracle top-2 margin < 0.1 is a knife-edge tie (fp16 class), not a failure.
Run the fp16 mode first — any position fp16 also flips is an oracle-resolution
artifact, not quantization damage.

Modes reuse the EXACT quant configs of the export script (imported from
conversion/export_qwen3_5_decode_pipelined.py), so the gated eager model and
the exported bundle share the recipe byte-for-byte.

Run:  cd coreai-models && .venv/bin/python _smoke/test_ornith9b_eager_gate.py \
          [fp16|int8lin|int8hu|int4lin]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

# Works from either home: coreai-models/_smoke/ (workspace sibling) or this repo's _smoke/.
_C1 = Path(__file__).resolve().parents[1] / "conversion" / "export_qwen3_5_decode_pipelined.py"
_C2 = Path(__file__).resolve().parents[2] / \
    "coreai-models-community" / "conversion" / "export_qwen3_5_decode_pipelined.py"
CONVERSION = _C1 if _C1.exists() else _C2
HF_ID = os.environ.get("ORNITH_HF_ID", "deepreinforce-ai/Ornith-1.0-9B")
REF = os.environ.get("ORNITH_REF", os.path.join(os.path.dirname(__file__), "ornith9b_ref.pt"))
N_DECODE = 8  # teacher-forced steps along the oracle greedy continuation


def load_export_module():
    spec = importlib.util.spec_from_file_location("qwen35_export", CONVERSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8hu",
                    choices=["fp16", "int8lin", "int8hu", "int4lin"])
    args = ap.parse_args()

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.models.macos.qwen3_5 import (
        Qwen3_5StatefulForCausalLM,
        build_decode_state,
    )
    from coreai_models.primitives.macos.cache import KVCache

    ref = torch.load(REF, weights_only=False)
    ids = ref["ids"].to(torch.int32)
    n = ids.shape[1]
    sweep = ref["sweep_positions"]
    print(f"oracle: {REF} ({n} prompt tokens, sweep {len(sweep)}, "
          f"decode {min(N_DECODE, len(ref['greedy_ids']))})")

    print(f"loading {HF_ID} fp16 eager CPU ...", flush=True)
    model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        HF_ID, max_context_length=4096, target_dtype=torch.float16,
        hf_config_attr="text_config")
    model.eval()
    cfg = model.config
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True

    if args.mode != "fp16":
        exp = load_export_module()
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = exp.linear_quant_config("int4" if args.mode == "int4lin" else "int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {
                r".*lm_head$": exp.head_quant_spec("block32", True)}  # ship shape
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())

        trace_past = 64
        state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN,
                                   dtype=torch.float16)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=4095)
        k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=4096)
        v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=4096)
        dynamic_shapes = {
            "input_ids": None,
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        print(f"quantizing eager ({args.mode}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # Teacher-forced S=1 sweep over the prompt + decode steps along the oracle
    # greedy continuation (the engine's exact per-token semantics).
    full = torch.cat(
        [ids, torch.tensor([ref["greedy_ids"][:N_DECODE]], dtype=torch.int32)], dim=1)
    st = build_decode_state(cfg, max_seq_len=full.shape[1] + 8, dtype=torch.float16)

    flips_conf, ok = [], True
    n_sweep_pass = n_dec_pass = 0
    for t in range(full.shape[1]):
        pos = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
        out = model(full[:, t : t + 1], pos, st["k_cache"], st["v_cache"],
                    st["conv_state"], st["rec_state"])
        a = int(out[0, 0].float().argmax())
        if t in sweep:
            want, m = int(ref["top1"][t]), float(ref["margin"][t])
            hit = a == want
            n_sweep_pass += hit or m < 0.1
            tag = "OK" if hit else ("tie(<0.1)" if m < 0.1 else "FLIP")
            if not hit and m >= 0.1:
                flips_conf.append((t, want, a, m))
                ok = False
            print(f"sweep pos {t:3d}: want {want:6d} got {a:6d} m={m:6.3f} {tag}")
        elif t >= n - 1:
            i = t - (n - 1)
            if i >= N_DECODE:
                break
            want, m = int(ref["greedy_ids"][i]), float(ref["greedy_margin"][i])
            hit = a == want
            n_dec_pass += hit or m < 0.1
            tag = "OK" if hit else ("tie(<0.1)" if m < 0.1 else "FLIP")
            if not hit and m >= 0.1:
                flips_conf.append((t, want, a, m))
                ok = False
            print(f"decode step {i:2d}: want {want:6d} got {a:6d} m={m:6.3f} {tag}")

    print(f"\n[{args.mode}] sweep {n_sweep_pass}/{len(sweep)} + "
          f"decode {n_dec_pass}/{N_DECODE} under margin>=0.1 rule")
    if flips_conf:
        print("confident flips:", flips_conf)
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
