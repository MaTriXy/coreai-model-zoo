# Community port — NOT an Apple model.
"""Export the Gemma-4 E2B MOBILE MIXED-BIT QAT TRANSPLANT as a pipelined decode bundle.

Google's mobile QAT weights (extracted bit-exact from `litert-community/gemma-4-E2B-it-litert-lm`,
recipe + verification in ../knowledge/gemma4-mixedbit-qat-transplant.md) transplanted into the
Core AI gemma4 pipelined graph. Per-channel PURE SYMMETRIC everywhere (zp=0):

  * FFN L0-14  INT4 x 6144-wide  -> affine int4 Metal kernel (exact sym->affine mapping)
  * FFN L15-34 INT2 x 12288-wide -> NEW int2sym Metal kernel
  * attn q/o (all 35)  INT4      -> affine int4 Metal kernel
  * attn k/v (L0-14)   INT4      -> dequantized fp16 (small-N; oracle-equivalent numerics)
  * lm_head (tied)     INT2      -> int2sym Metal kernel, N=262144
  * embed_tokens       INT2      -> in-graph packed table + byte-LUT gather (PackedInt2Embedding)
  * PLE table (35xINT4)          -> in-graph packed table + byte-LUT gather (no static inputs!)
  * PLE gate/proj + model_proj INT8 -> shipped int8 per-block-32 requant (near-lossless)
  * all norms / layer scalars    -> the extracted fp32 QAT values (NOT the public checkpoint)

Active decode read ~780 MB/token vs the shipped int4lin E2B's ~2.0 GB — the byte-parity thesis
vs LiteRT-LM (Mac decode 113.2, iPhone 30.8 tok/s).

No extra inputs: both tables ride as in-graph constants, so the bundle runs on stock
llm-benchmark / PipelinedBench with no staticInputBuffers binding.

Run (from a coreai-models checkout with the model overlay):
  .venv/bin/python ../coreai-models-community/conversion/export_gemma4_mixedbit_decode_pipelined.py
Gate next: the mixedbit greedy gate (engine vs the MLX transplant oracle), then
  COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model exports/<name> -p 128 -g 256 -n 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from _paths import code_path

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.gemma4_metal_mlp_int2 import (
    MetalInt2SymLinear,
    MetalInt2SymMLP,
    MetalInt2SymMLPFused,
    MetalInt4AffLinear,
    MetalInt4AffMLP,
    MetalInt4AffMLPFused,
    build_fused_int2sym_kernel,
    build_fused_int4_kernel,
    build_gateup_int2sym_kernel,
    build_gateup_int4aff_kernel,
    unpack_int4_codes,
)
from coreai_models.models.macos.gemma4_mixedbit_pipelined import (
    Gemma4MixedbitPipelinedForCausalLM,
    PackedInt2Embedding,
)
from coreai_models.models.macos.gemma4_text import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    _truncated_kv_shared,
)

DEFAULT_EXTRACT = str(code_path("litertlm-convert", "out", "gemma4e2b_extract"))
DEFAULT_HF_ID = "google/gemma-4-E2B-it"  # config + tokenizer only; NO weights are read from HF
# Provenance for the mixed-bit weights. Google published the wNa8o8 mobile QAT run as a plain
# Apache-2.0 Transformers checkpoint on 2026-07-15; it is bit-exact with the older
# .litertlm extraction (see knowledge/gemma4-litertlm-to-official-migration.md), so the
# official checkpoint is the source of record and nothing here depends on cracking a binary.
DEFAULT_WEIGHTS_SOURCE = "google/gemma-4-E2B-it-qat-mobile-transformers (official wNa8o8 mobile QAT)"
DTYPE = torch.float16
FIRST_SHARED = 15


class Extract:
    """The P1 extraction artifacts (packed codes + per-row fp32 scales + fp32 norms)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.w = safe_open(str(root / "gemma4e2b_mixedbit_weights.safetensors"), framework="pt")
        self.n = safe_open(str(root / "gemma4e2b_fp32_norms.safetensors"), framework="pt")
        self.manifest = json.loads((root / "gemma4e2b_mixedbit_manifest.json").read_text())

    def packed(self, key: str, bits: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        m = self.manifest[key]
        assert m["bits"] == bits, (key, m["bits"], bits)
        rows, cols = m["shape"]
        return self.w.get_tensor(key), self.w.get_tensor(key + ".scale").float(), rows, cols

    def dequant_fp(self, key: str) -> torch.Tensor:
        """Small-tensor fp dequant (int4/int8) — the oracle's convention, fp32."""
        m = self.manifest[key]
        rows, cols = m["shape"]
        packed = self.w.get_tensor(key)
        scale = self.w.get_tensor(key + ".scale").float()
        if m["bits"] == 8:
            c = packed.view(torch.int8).reshape(rows, cols).float()
        elif m["bits"] == 4:
            c = unpack_int4_codes(packed, rows, cols).float()
        else:
            raise ValueError((key, m["bits"]))
        return c * scale.unsqueeze(1)

    def norm(self, key: str) -> torch.Tensor:
        return self.n.get_tensor(key).float()


def load_transplant_weights(causal: Gemma4ForCausalLM, ex: Extract) -> None:
    """Fill every surviving nn parameter from the extraction (norms, scalars, small fp16 mats)."""
    cfg = causal.config
    m = causal.model
    m.norm.weight.data = torch.from_numpy(
        np.load(ex.root / "final_norm.f32.npy")).to(DTYPE)
    m.per_layer_projection_norm.weight.data = ex.norm("ple_projection_norm").to(DTYPE)
    m.per_layer_model_projection.weight.data = (
        ex.dequant_fp("decode.ple.model_proj")[: m.per_layer_model_projection.weight.shape[0]]
        .to(DTYPE))

    for li, layer in enumerate(m.layers):
        N, Q = f"layer_{li:02d}.", f"decode.layer_{li:02d}."
        layer.input_layernorm.weight.data = ex.norm(N + "pre_attention_norm").to(DTYPE)
        layer.post_attention_layernorm.weight.data = ex.norm(N + "post_attention_norm").to(DTYPE)
        layer.pre_feedforward_layernorm.weight.data = ex.norm(N + "pre_ffw_norm").to(DTYPE)
        layer.post_feedforward_layernorm.weight.data = ex.norm(N + "post_ffw_norm").to(DTYPE)
        layer.post_per_layer_input_norm.weight.data = (
            ex.norm(N + "post_per_layer_input_norm").to(DTYPE))
        layer.layer_scalar.data = ex.norm(N + "skip_scale").reshape(1).to(DTYPE)
        attn = layer.self_attn
        attn.q_norm.weight.data = ex.norm(N + "query_norm").to(DTYPE)
        if li < FIRST_SHARED:
            attn.k_norm.weight.data = ex.norm(N + "key_norm").to(DTYPE)
            attn.k_proj.weight.data = ex.dequant_fp(Q + "attn.k").to(DTYPE)
            attn.v_proj.weight.data = ex.dequant_fp(Q + "attn.v").to(DTYPE)
        else:
            # Dead weights: KV-shared layers never project K/V (they read the producer slot).
            attn.k_norm.weight.data.fill_(1.0)
            attn.k_proj.weight.data.zero_()
            attn.v_proj.weight.data.zero_()
        layer.per_layer_input_gate.weight.data = ex.dequant_fp(Q + "ple.gate").to(DTYPE)
        layer.per_layer_projection.weight.data = ex.dequant_fp(Q + "ple.proj").to(DTYPE)


def metalize_transplant(model, ex: Extract, k2, k4, gu2=None, gu4=None) -> int:
    """Swap FFN / attn q/o / lm_head for the pre-packed transplant kernel modules.

    With ``gu2``/``gu4`` (fused gate+up+gelu+mul kernels) the FFN runs in TWO dispatches
    per layer instead of ~five (the Mac dispatch-floor lever)."""
    n = 0
    for li, layer in enumerate(model.model.layers):
        Q = f"decode.layer_{li:02d}."
        bits = ex.manifest[Q + "mlp.gating1"]["bits"]
        packed = {
            "gate": ex.packed(Q + "mlp.gating1", bits),
            "up": ex.packed(Q + "mlp.gating2", bits),
            "down": ex.packed(Q + "mlp.down", bits),
        }
        if bits == 2:
            layer.mlp = (MetalInt2SymMLPFused(packed, gu2, k2) if gu2 is not None
                         else MetalInt2SymMLP(packed, k2))
        else:
            layer.mlp = (MetalInt4AffMLPFused(packed, gu4, k4) if gu4 is not None
                         else MetalInt4AffMLP(packed, k4))
        attn = layer.self_attn
        attn.q_proj = MetalInt4AffLinear(*ex.packed(Q + "attn.q", 4), k4)
        attn.o_proj = MetalInt4AffLinear(*ex.packed(Q + "attn.o", 4), k4)
        n += 3
    model.lm_head = MetalInt2SymLinear(*ex.packed("decode.lm_head", 2), k2)
    return n + 1


def int8_requant_config() -> dict:
    """Shipped int8 linear per-block-32 for the LiteRT-INT8 tensors ONLY (PLE projections).

    Everything headed for a transplant kernel (FFN, attn q/o, lm_head) plus the fp16-kept
    attn k/v is excluded. Requantizing the dequantized per-channel-int8 QAT values onto the
    per-block grid is near-lossless (the values already sit on an int8 grid per row).
    """
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int8",
                    "qscheme": "symmetric",
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
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {
            r".*lm_head$": None,
            r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)$": None,
            r".*mlp\.(gate_proj|up_proj|down_proj)$": None,
        },
    }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, cfg, max_ctx: int,
                          weights_source: str) -> None:
    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": hf_id,
            "vocab_size": cfg.vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_id,
                   "weights": weights_source},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract-dir", default=DEFAULT_EXTRACT)
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID, help="config + tokenizer source (no weights)")
    # Recorded verbatim in metadata.json. The extract dir carries no provenance of its own, so
    # this must name whatever produced --extract-dir. Default = the official Apache-2.0 mobile QAT
    # checkpoint; pass the litert-community string only for a bundle built from the old
    # reverse-engineered .litertlm extraction.
    ap.add_argument("--weights-source", default=DEFAULT_WEIGHTS_SOURCE,
                    help="provenance string written to metadata.json source.weights")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--fused-ffn", action="store_true",
                    help="FFN first half (gate+up+gelu+mul) as ONE kernel dispatch per layer")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(args.hf_id, allow_patterns=["*.json"])
    d = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = Gemma4TextConfig.from_hf_config(d)
    if args.num_layers is not None:
        cfg.num_kv_shared_layers = _truncated_kv_shared(cfg, args.num_layers)
        cfg.num_hidden_layers = args.num_layers
        cfg.layer_types = cfg.layer_types[: args.num_layers]
    L = cfg.num_hidden_layers
    name = ("gemma4_e2b_mixedbit_decode" + ("_ffnfused" if args.fused_ffn else "")
            + (f"_l{L}" if args.num_layers is not None else ""))

    ex = Extract(Path(args.extract_dir))

    # Construct with a stub vocab so the (replaced-anyway) embed/lm_head/PLE tables don't pay
    # a multi-GB random init; restore the real vocab before building the export spec.
    real_v, real_vp = cfg.vocab_size, cfg.vocab_size_per_layer_input
    cfg.vocab_size = cfg.vocab_size_per_layer_input = 64
    causal = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    cfg.vocab_size, cfg.vocab_size_per_layer_input = real_v, real_vp
    del causal.model.embed_tokens_per_layer

    print(f"loading transplant weights (L={L}) ...", flush=True)
    load_transplant_weights(causal, ex)

    # In-graph INT2 embed table (tied source of the lm_head, gathered packed).
    emb_packed, emb_scale, emb_rows, emb_cols = ex.packed("embed.composite", 2)
    assert (emb_rows, emb_cols) == (real_v, cfg.hidden_size)
    causal.model.embed_tokens = PackedInt2Embedding(
        emb_packed, emb_scale, cfg.hidden_size, embed_scale=cfg.hidden_size**0.5)
    causal.lm_head = torch.nn.Linear(cfg.hidden_size, 64, bias=False).to(DTYPE)  # placeholder

    # In-graph INT4 PLE table: concat the 35 per-layer tables in PACKED byte space (each table's
    # 256 int4 codes = 128 bytes, so byte-concat == code-concat) + per (row, table) scales.
    print("assembling packed PLE table ...", flush=True)
    tables, scales = [], []
    for i in range(L):
        key = "ple_table.composite" + ("" if i == 0 else str(i))
        p, s, rows, cols = ex.packed(key, 4)
        assert (rows, cols) == (real_vp, cfg.hidden_size_per_layer_input)
        tables.append(p.reshape(rows, cols // 2))
        scales.append(s)
    ple_packed = torch.cat(tables, dim=1).contiguous()
    ple_scale = torch.stack(scales, dim=1).contiguous()
    del tables, scales

    model = Gemma4MixedbitPipelinedForCausalLM(causal, ple_packed, ple_scale).eval()
    spec = model.build_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    from coreai_models.export.compression import quantize_pytorch_model

    print("quantizing PLE projections (shipped int8 per-block-32; all else excluded) ...",
          flush=True)
    model = quantize_pytorch_model(
        model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"],
        int8_requant_config())

    print("metalizing transplant kernels (int2sym + int4 affine"
          + (" + fused gateup" if args.fused_ffn else "") + ") ...", flush=True)
    k2 = build_fused_int2sym_kernel()
    k4 = build_fused_int4_kernel()
    kernels = [k2, k4]
    gu2 = gu4 = None
    if args.fused_ffn:
        gu2 = build_gateup_int2sym_kernel()
        gu4 = build_gateup_int4aff_kernel()
        kernels += [gu2, gu4]
    n = metalize_transplant(model, ex, k2, k4, gu2=gu2, gu4=gu4)
    print(f"metalized {n} matvec sites (FFN + attn q/o + lm_head)", flush=True)

    print("exporting decode-only engine graph (in-graph packed tables + 2 kernels) ...",
          flush=True)
    prog = export_to_coreai_with_kernels(
        model,
        reference_inputs=spec["reference_inputs"],
        custom_kernels=kernels,
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=spec["state_names"],
        externalize_modules=(),  # gemma4 opts out (orphan PLE front-end norms)
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx, args.weights_source)
    tok_src = Path(snapshot_download(
        args.hf_id, allow_patterns=["tokenizer*", "special_tokens_map.json"]))
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_src / f).exists():
            shutil.copy(tok_src / f, tok_dir / f)

    import subprocess

    sz = subprocess.run(["du", "-sh", str(out_dir)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle ready: {out_dir} ({sz})")
    print("gate next: mixedbit greedy gate (engine vs MLX transplant oracle), then\n"
          f"  COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
