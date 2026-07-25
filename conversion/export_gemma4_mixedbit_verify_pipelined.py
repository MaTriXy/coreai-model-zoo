# Community port — NOT an Apple model.
"""Export the S=4 VERIFY companion of the Gemma-4 E2B mixed-bit transplant.

Weight-identical to the shipped mixedbit decode bundle (same extraction, same
recipes) but: static [1,4] query, M=4 transplant kernels (weights read once for
4 tokens), and an extra `activations` output (final-norm hidden) — the MTP
drafter's kickoff/chaining tap. Contract details in gemma4_mixedbit_verify.py.

Run (coreai-models checkout):
  .venv/bin/python ../coreai-models-community/conversion/export_gemma4_mixedbit_verify_pipelined.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.gemma4_metal_mlp_m4 import (
    M4Int2SymLinear,
    M4Int2SymMLPFused,
    M4Int4AffLinear,
    M4Int4AffMLPFused,
    build_gateup_int2sym_m4_kernel,
    build_gateup_int4aff_m4_kernel,
    build_int2sym_m4_kernel,
    build_int4aff_m4_kernel,
)
from coreai_models.models.macos.gemma4_mixedbit_pipelined import PackedInt2Embedding
from coreai_models.models.macos.gemma4_mixedbit_verify import (
    Gemma4MixedbitVerifyForCausalLM,
)

DTYPE = torch.float16

# reuse the decode export's Extract / config assembly / metadata machinery
_DEC = Path(__file__).parent / "export_gemma4_mixedbit_decode_pipelined.py"
_spec = importlib.util.spec_from_file_location("mixedbit_decode_export", _DEC)
_dec = importlib.util.module_from_spec(_spec)
sys.modules["mixedbit_decode_export"] = _dec
_spec.loader.exec_module(_dec)


def metalize_transplant_m4(model, ex, k2, k4, gu2, gu4) -> int:
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
            layer.mlp = M4Int2SymMLPFused(packed, gu2, k2)
        else:
            layer.mlp = M4Int4AffMLPFused(packed, gu4, k4)
        attn = layer.self_attn
        attn.q_proj = M4Int4AffLinear(*ex.packed(Q + "attn.q", 4), k4)
        attn.o_proj = M4Int4AffLinear(*ex.packed(Q + "attn.o", 4), k4)
        n += 3
    model.lm_head = M4Int2SymLinear(*ex.packed("decode.lm_head", 2), k2)
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default=_dec.DEFAULT_EXTRACT)
    ap.add_argument("--hf-id", default=_dec.DEFAULT_HF_ID)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()

    name = "gemma4_e2b_mixedbit_verify_s4"
    ex = _dec.Extract(Path(args.extract))

    from huggingface_hub import snapshot_download

    from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLM, Gemma4TextConfig

    model_dir = snapshot_download(args.hf_id, allow_patterns=["*.json"])
    d = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = Gemma4TextConfig.from_hf_config(d)
    # stub vocab during construction so the replaced-anyway tables skip the giant init
    real_v, real_vp = cfg.vocab_size, cfg.vocab_size_per_layer_input
    cfg.vocab_size = cfg.vocab_size_per_layer_input = 64
    causal = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    cfg.vocab_size, cfg.vocab_size_per_layer_input = real_v, real_vp
    del causal.model.embed_tokens_per_layer
    print("loading transplant weights ...", flush=True)
    _dec.load_transplant_weights(causal, ex)

    emb_packed, emb_scale, emb_rows, emb_cols = ex.packed("embed.composite", 2)
    causal.model.embed_tokens = PackedInt2Embedding(
        emb_packed, emb_scale, cfg.hidden_size, embed_scale=cfg.hidden_size**0.5)
    causal.lm_head = torch.nn.Linear(cfg.hidden_size, 64, bias=False).to(DTYPE)

    print("assembling packed PLE table ...", flush=True)
    tables, scales = [], []
    for i in range(cfg.num_hidden_layers):
        key = "ple_table.composite" + ("" if i == 0 else str(i))
        p, s, rows, cols = ex.packed(key, 4)
        tables.append(p.reshape(rows, cols // 2))
        scales.append(s)
    ple_packed = torch.cat(tables, dim=1).contiguous()
    ple_scale = torch.stack(scales, dim=1).contiguous()
    del tables, scales

    model = Gemma4MixedbitVerifyForCausalLM(causal, ple_packed, ple_scale).eval()
    spec = model.build_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    from coreai_models.export.compression import quantize_pytorch_model

    print("quantizing PLE projections (shipped int8 per-block-32) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"],
        _dec.int8_requant_config())

    print("metalizing with M=4 transplant kernels ...", flush=True)
    k2 = build_int2sym_m4_kernel()
    k4 = build_int4aff_m4_kernel()
    gu2 = build_gateup_int2sym_m4_kernel()
    gu4 = build_gateup_int4aff_m4_kernel()
    n = metalize_transplant_m4(model, ex, k2, k4, gu2, gu4)
    print(f"metalized {n} matvec sites (M=4)", flush=True)

    print("exporting S=4 verify graph ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model,
        reference_inputs=spec["reference_inputs"],
        custom_kernels=[k2, k4, gu2, gu4],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=spec["state_names"],
        externalize_modules=(),
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    _dec.write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx)
    meta = json.loads((out_dir / "metadata.json").read_text())
    meta["verify_query_len"] = 4
    meta["compilation"]["date"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
