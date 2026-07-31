"""Export a DECODE-ONLY GLM-4.7-Flash bundle with ABSORBED-MLA attention.

Absorbed-MLA caches only the [512] kv-latent + [64] rope key (vs the naive
materialized per-head K/V) and folds the kv_b up-projection into the q-lift /
latent-value readout — ~17.8× smaller KV and no per-token kv_b read (see
``ABSORBED_MLA_STATE.md``). The MoE side keeps the shipped ``gather_qmm`` sym8
kernel (reads only the 4/64 routed experts). Two MLA backends:

  * default (graph probe): the eager einsum attention core lowers to MPSGraph
    matmul+softmax. Use this to check whether the 576-dim absorbed shape heap-crashes
    or is merely slow on the engine.
  * ``--metal-sdpa``: swap the core for the absorbed-MLA flash-decode Metal kernel
    (``mla_metal_sdpa.py``) — the fused, online-softmax decode (bundle suffix ``_msdpa``).

Everything outside the attention + routed experts is the shipped int8 recipe
(int8 per-block-32 linears incl. the MLA projections + shared expert; fp16 router;
absmax int8 head). W_UK/W_UV (the small absorbed lifts) stay fp16.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_glm47_absorbed_decode_pipelined.py \
          --hf-id zai-org/GLM-4.7-Flash [--metal-sdpa] [--split-g 8] [--num-layers N]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.glm4_moe_lite_absorbed import (
    DECODE_STATE_NAMES_ABSORBED,
    Glm4MoeLiteAbsorbedStatefulForCausalLM,
    build_absorbed_decode_state,
    glm4_moe_lite_absorbed_from_hf,
)
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """int8 per-block-32 for the dense linears (MLA q_a/q_b/kv_a/o_proj, shared expert).
    Routed experts (SwitchLinear) -> None (the gather_qmm kernel owns them). W_UK/W_UV are
    buffers, not Linear -> untouched (stay fp16)."""
    block_2d = {"dtype": dtype, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}
    return {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": block_2d}, "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
            "coreai_models.primitives.macos.switch.SwitchLinear": None,  # routed experts -> metal
        },
        "module_name_configs": {r".*mlp\.gate$": None, r".*lm_head$": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--metal-sdpa", action="store_true",
                    help="use the absorbed-MLA flash-decode kernel (else eager einsum graph)")
    ap.add_argument("--split-g", type=int, default=32,
                    help="staged MLA kernel sequence-split factor (# threadgroups for occupancy)")
    ap.add_argument("--per-head", action="store_true",
                    help="use the non-staging per-head baseline kernel (A/B; default = staged)")
    ap.add_argument("--head-quant", default="block32")
    ap.add_argument("--head-sym", action="store_true")
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_absorbed" + ("_msdpa" if args.metal_sdpa else "_graph")
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 (absorbed) ...", flush=True)
    causal = glm4_moe_lite_absorbed_from_hf(args.hf_id, target_dtype=DTYPE)
    if args.num_layers is not None:
        causal.model.layers = causal.model.layers[: args.num_layers]
        causal.config.num_hidden_layers = args.num_layers
    model = Glm4MoeLiteAbsorbedStatefulForCausalLM.from_causal_lm(causal)
    model.eval()
    cfg = causal.config
    print(f"model ready | {cfg.num_hidden_layers} layers, E={cfg.n_routed_experts}/"
          f"top{cfg.num_experts_per_tok}, kv_lora={cfg.kv_lora_rank}, vocab={cfg.vocab_size}", flush=True)

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_absorbed_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "kv_a": state["kv_a"], "kv_b": state["kv_b"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    a_seq = torch.export.Dim("a_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    b_seq = torch.export.Dim("b_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos},
        "kv_a": {KVCache.seq_len_dim(): a_seq},
        "kv_b": {KVCache.seq_len_dim(): b_seq},
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) int8 the dense linears (routed experts excluded -> stay fp16 for metalize)
    from coreai_models.export.compression import quantize_pytorch_model
    cfg_q = linear_quant_config("int8")
    cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(args.head_quant, args.head_sym)
    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing dense linears (int8 per-block-32; routed experts kept fp16) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) metalize the routed MoE (gather_qmm sym8 — reads only top-4/64)
    print("metalizing routed MoE -> gather_qmm sym8 ...", flush=True)
    kernels = [metalize_moe(model, scheme="sym8")]

    # 3) optionally swap the MLA core for the absorbed flash-decode kernel
    if args.metal_sdpa:
        from coreai_models.models.macos.mla_metal_sdpa import metalize_mla
        staged = not args.per_head
        kind = f"staged (split_g={args.split_g})" if staged else "per-head baseline"
        print(f"metalizing MLA -> absorbed flash-decode kernel: {kind} ...", flush=True)
        kernels.extend(metalize_mla(causal, scale=cfg.qk_head_dim ** -0.5,
                                    staged=staged, split_g=args.split_g))
    else:
        print("MLA kept as eager einsum graph (probe mode) ...", flush=True)

    # 4) export with the kernel(s); drop gather_mm + gated_delta_update specs
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs=reference_inputs, custom_kernels=kernels,
        dynamic_shapes=dynamic_shapes, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES_ABSORBED, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
