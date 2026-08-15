"""Eager quant-recipe gate for Qwen3.8-27B: overlay (fake-quant) vs the HF bf16 oracle.

Teacher-forced single-step argmax, margin>=0.1 rule — the ONLY reliable gate for a
multi-GB graph (NEVER gate via raw AIModel.load(...gpu()); its fp16/ANE fallback
returns garbage that does not reflect engine numerics — 3.6-27B session, hard-won).

Modes:
  fp16    full-precision control (attributes oracle-resolution flips: a position that
          fp16 ALSO flips, byte-identical, is a bf16-oracle artifact, not a quant defect)
  int8hu  the ship recipe: int8 per-block-32 symmetric-with-clipping body + absmax
          symmetric int8 block32 untied head (== export int8hu --head-sym)
  int4lin int4 per-block-32 body, head fp16 (the speed option; borderline on 3.6-27B)

Prereq: _smoke/qwen38_27b_ref.pt from gen_qwen38_27b_ref.py (oracle venv).
Runs in the EXPORT venv (coreai-models/.venv): needs the qwen3_5 overlay + coreai-opt.

Usage: cd coreai-models-community && \
       ../coreai-models/.venv/bin/python _smoke/test_qwen38_27b_int8_eager_gate.py int8hu
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "conversion"))

from _bundle import head_quant_spec  # noqa: E402

HF_ID = "Qwen/Qwen3.8-27B"
REF = HERE / "qwen38_27b_ref.pt"
MARGIN = 0.1
DTYPE = torch.float16
CTX = 4096


def linear_quant_config(dtype: str = "int8") -> dict:
    # Verbatim from conversion/export_qwen3_5_decode_pipelined.py — the gate must
    # quantize EXACTLY what the export quantizes.
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "int8hu"
    assert mode in ("fp16", "int8hu", "int4lin"), mode
    ref = torch.load(REF, weights_only=False)
    prompt_ids = ref["prompt_ids"].to(torch.int32).unsqueeze(0)
    gen_ids = ref["gen_ids"].tolist()
    o_margins = ref["margins"].tolist()
    o_logits = ref["logits_rows"].float()
    n = len(gen_ids)
    print(f"oracle: {n} positions, prompt len {prompt_ids.shape[1]}, "
          f"continuation {gen_ids}")

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.models.macos.qwen3_5 import (
        Qwen3_5StatefulForCausalLM,
        build_decode_state,
    )
    from coreai_models.primitives.macos.cache import KVCache

    print(f"loading {HF_ID} fp16 overlay ...")
    model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        HF_ID, max_context_length=CTX, target_dtype=DTYPE, hf_config_attr="text_config"
    )
    model.eval()
    cfg = model.config
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True

    if mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        trace_past = 64
        state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=CTX - 1)
        k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=CTX)
        v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=CTX)
        dynamic_shapes = {
            "input_ids": None,
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        cfg_q = linear_quant_config("int4" if mode == "int4lin" else "int8")
        if mode == "int8hu":
            cfg_q["module_name_configs"] = {
                r".*lm_head$": head_quant_spec("block32", sym=True)
            }
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone()
            )
        print(f"fake-quantizing ({mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q
        )

    st = build_decode_state(cfg, max_seq_len=CTX, dtype=DTYPE)
    states = [st[k] for k in ("k_cache", "v_cache", "conv_state", "rec_state")]

    # Teacher-forced walk: feed prompt then the ORACLE's tokens; compare our argmax
    # at each continuation position. Full-range position_ids each step (S=1 idiom).
    seq = torch.cat([prompt_ids, torch.tensor([gen_ids], dtype=torch.int32)], dim=1)
    plen = prompt_ids.shape[1]
    n_pass = n_flip_tie = 0
    failures = []
    with torch.no_grad():
        for t in range(seq.shape[1] - 1):
            out = model(
                seq[:, t : t + 1],
                torch.arange(t + 1, dtype=torch.int32).unsqueeze(0),
                *states,
            )
            if t < plen - 1:
                continue
            k = t - (plen - 1)  # continuation position index
            row = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].float()
            ours = int(row.argmax())
            want = gen_ids[k]
            cos = float(
                torch.nn.functional.cosine_similarity(row, o_logits[k], dim=0)
            )
            if ours == want:
                n_pass += 1
                verdict = "PASS"
            elif o_margins[k] < MARGIN:
                n_flip_tie += 1
                verdict = f"tie-flip (margin {o_margins[k]:.3f} < {MARGIN}) — passes rule"
            else:
                failures.append((k, ours, want, o_margins[k], cos))
                verdict = f"FAIL confident flip (margin {o_margins[k]:.3f})"
            print(f"pos {k:2d}: ours={ours:6d} oracle={want:6d} "
                  f"margin={o_margins[k]:.3f} cos={cos:.5f}  {verdict}")

    print(f"\n[{mode}] exact {n_pass}/{n} | tie-flips {n_flip_tie} | "
          f"confident flips {len(failures)}")
    if failures:
        print("CONFIDENT FLIPS:", failures)
        print("GATE: FAIL")
        sys.exit(1)
    print("GATE: PASS (margin>=%.1f rule)" % MARGIN)


if __name__ == "__main__":
    main()
