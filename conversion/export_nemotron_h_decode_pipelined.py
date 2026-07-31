"""Export a DECODE-ONLY (S=1) NVIDIA-Nemotron-3-Nano-4B bundle for the Core AI pipelined engine.

Nemotron-H (NVIDIA) is a Mamba2 + attention + MLP hybrid — the zoo's second SSM-scan
architecture after Granite 4.0-H, and the first that is not IBM's. The 4B's
`hybrid_override_pattern` gives 42 blocks: **21 Mamba2 + 17 MLP + 4 GQA NoPE attention**,
one mixer per block (a mamba block carries no second MLP branch). At S=1 the Mamba2
selective scan collapses to a single recurrence step (`state = state*dA + dt*B*x`), so the
decode graph is loop-free and lowers on the MPSGraph GPU delegate — the same reason
granite4h and qwen3.5's GDN do. State = growing KV (4 attn layers) + TWO fixed-shape extra
states (stacked conv columns + stacked SSM states), exactly the
coreai-pipelined-extra-states.patch budget (<=2).

input_ids is STATIC [1,1]; position_ids and the KV seq dim stay dynamic, so `EngineFactory`
classifies the bundle as dynamic -> pipelined engine. Prefill runs as pipelined S=1 steps:
set `COREAI_CHUNK_THRESHOLD=1`.

A 4B graph cannot specialize on-device, so the iPhone bundle must be AOT-compiled:
    xcrun coreai-build compile <out>/<name>.aimodel \
        --platform iOS --preferred-compute gpu --architecture h18p --output <out>
then rewrite `metadata.json` "assets.main" to the compiled `<name>.h18p.aimodelc`
(CoreAIShared.ModelBundle reads metadata.json at the model dir, not inside the .aimodelc).

Numerics gate: the port is token-identical to the fp32 HF rollout (per-step logits
rel 2e-7…4e-7 over the prompt, 8 greedy tokens exact), and the exported int8hu bundle
reproduces the fp32 oracle's top-1 at a margin-clean position on the GPU. On device:
nat 24/24 + oracle 24/24 (PipelinedBench).

Modes: fp16 - baseline (7.6 GiB, Mac only); int8lin - per-block-32 linear int8 body,
fp16 head (4.7 GiB); int8hu - int8lin + absmax int8 lm_head (4.3 GiB, the SHIP config —
the head is untied here and a fat-tailed 131k-vocab head needs absmax, not clipping).

**4-bit does not ship for this model, and the reason is not quality alone** (see
`models/nemotron-3-nano/README.md`): symmetric int4 dequants cheaper than int8 but flips 4-6 of the
33 margin-clean oracle positions at every block size; asymmetric int4 at block-16 is the
only 4-bit scheme that gates 33/33, yet its zero-point dequant costs ~0.42 ms/layer and
lands at 3.0-3.5 tok/s on device (4.6x slower than int8) despite reading 1.45x fewer bytes;
and the k-means format the fused `int4km` kernel reads is the worst of the three (22/33) —
its codebook has no scale along K — while `hidden_size = 3136 = 256*12 + 64` makes 65% of
the weights ineligible for that kernel's `K % 256 == 0` shape constraint anyway.

Requires the nemotron_h model overlay on `coreai-models` (see conversion/README.md) plus the
pipelined-engine extra-states patch on the Swift side to RUN the bundle.

Run:  python export_nemotron_h_decode_pipelined.py [fp16|int8lin|int8hu] \
          [--hf-id nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16] [--out-dir exports]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.nemotron_h import (
    DECODE_STATE_NAMES,
    NemotronHForCausalLMStateful,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 — scale-multiply dequant, no LUT.
    Embedding / Conv1d / norms (incl. the grouped gated Mamba output norm) excluded by
    type; the head excluded by name unless int8hu adds it back below."""
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
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.models.macos.nemotron_h.NemotronHGatedRMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--hf-id", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"],
                    help="int8hu only: lm_head weight granularity (ship=block32)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: plain symmetric (absmax, no clipping) for the head")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"

    print(f"loading {args.hf_id} fp16 ...")
    model = NemotronHForCausalLMStateful.from_hf(args.hf_id, target_dtype=DTYPE).eval()
    cfg = model.config
    n_mlp = cfg.num_hidden_layers - cfg.num_mamba_layers - cfg.num_attn_layers
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_mamba_layers} mamba / "
          f"{cfg.num_attn_layers} attention / {n_mlp} mlp), hidden={cfg.hidden_size}, "
          f"vocab={cfg.vocab_size}")

    # Decode trace: S=1 static query, dynamic full-length positions, dynamic KV seq.
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
        "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1] — single recurrence step, no scan
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,  # fixed-shape extra states
        "rec_state": None,
    }

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config()
        if args.mode == "int8hu":
            # Nemotron-H's head is already untied, so unlike granite there is nothing to
            # clone first.
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                args.head_quant, args.head_sym)
        print(f"quantizing (linear int8 per-block-32, mode={args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # Nemotron-H calls neither GatedDeltaUpdate nor RoPE — drop those externalize specs, or
    # the exporter hunts for submodules that never run.
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "rope")]
    print("exporting decode-only graph to Core AI dialect ...")
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
                          mode=args.mode)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"iPhone (4B cannot specialize on-device — AOT is required):\n"
          f"  xcrun coreai-build compile {aimodel} --platform iOS "
          f"--preferred-compute gpu --architecture h18p --output {out_dir}\n"
          f"  then set metadata.json assets.main = {name}.h18p.aimodelc")


if __name__ == "__main__":
    main()
