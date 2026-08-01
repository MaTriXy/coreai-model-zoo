"""BitCPM-8B chunked-prefill bundle: `main` (S=1 ternary matvec) + `prefill` (S=C tiled GEMM).

The shipped `_s1` bundle walks a prompt one token at a time because the ternary kernel is a
matvec. This adds a second entrypoint traced at a static S=C that runs the tiled GEMM
(`bitcpm_ternary_gemm`), so the engine can feed the prompt in chunks. Weights are shared —
`export_to_coreai_multifunction` dedupes constants across entrypoints, so the second function
costs graph, not gigabytes.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_bitcpm8b_chunked_prefill.py --chunk 64
"""
from __future__ import annotations

import argparse, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

import coreai_torch
from _paths import work_path
from coreai_models.export._constants import (
    KEY_CACHE_NAME, TRACE_KV_CACHE_SEQ_LEN, VALUE_CACHE_NAME,
)
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from coreai_models.models.macos.bitcpm import load_bitcpm8b_from_gguf
from coreai_models.models.macos.bitcpm_ternary_gemm import build_tern_gemm_kernel
from coreai_models.models.macos.bitcpm_ternary_metal import (
    _R, _SGY, MetalTernaryLinear, build_tern_kernel,
)
from coreai_models.primitives.macos.cache import KVCache

GGUF = str(work_path("_bitcpm_ckpt", "bitcpm4-8b-tq2_0.gguf"))
HF = str(work_path("_bitcpm_ckpt", "hf"))
DTYPE = torch.float16
_GEMM_BN = 64


class DualTernaryLinear(nn.Module):
    """Same ternary weights, two kernels: matvec at S=1, tiled GEMM at S=chunk.

    `s` is a Python int under torch.export (both entrypoints are traced at a static query
    length), so the branch is resolved at trace time — each function gets exactly one kernel.
    """

    def __init__(self, src: MetalTernaryLinear, mv_kernel, gemm_kernel, chunk: int) -> None:
        super().__init__()
        self.N, self.chunk = src.N, chunk
        self.mv_kernel, self.gemm_kernel = mv_kernel, gemm_kernel
        self.register_buffer("qp", src.qp)
        self.register_buffer("d", src.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape
        if isinstance(s, int) and s != 1:
            y = self.gemm_kernel(x.reshape(s, k), self.qp, self.d,
                                 threads_per_grid=(32, 8 * (self.N // _GEMM_BN), 1),
                                 threads_per_thread_group=(32, 8, 1),
                                 result_shapes=[[s, self.N]])
        else:
            y = self.mv_kernel(x.reshape(s, k), self.qp, self.d,
                               threads_per_grid=(32, self.N // _R, 1),
                               threads_per_thread_group=(32, _SGY, 1),
                               result_shapes=[[s, self.N]])
        return y.reshape(b, s, self.N)


def swap_linears(model: nn.Module, mv_kernel, gemm_kernel, chunk: int) -> int:
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, MetalTernaryLinear):
                setattr(mod, name, DualTernaryLinear(child, mv_kernel, gemm_kernel, chunk))
                n += 1
    return n


def build_spec(cfg, max_ctx: int, query: int):
    """Static query length `query`, dynamic position/KV — the `_s1` contract widened to S=query."""
    input_ids = torch.randint(1, cfg.vocab_size, (1, query), dtype=torch.int32)
    # the traced position length must satisfy the Dim's own min (= query) or export rejects it
    position_ids = torch.arange(max(65, query + 1), dtype=torch.int32).unsqueeze(0)
    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved
    seq_dim = KVCache.seq_len_dim()
    return {
        "reference_inputs": {"input_ids": input_ids, "position_ids": position_ids,
                             "k_cache": k_cache, "v_cache": v_cache},
        "dynamic_shapes": {
            "input_ids": None,
            # min=query, NOT max(2,query): iOS MPSGraph asserts `Failed to resolve dynamic
            # dimensions for memref.alloc` when the S=1 entrypoint is driven at position 0
            # (length 1 < min 2). macOS tolerates it; the device does not.
            "position_ids": {1: torch.export.Dim("seq_pos", min=query, max=max_ctx - 1)},
            "k_cache": {seq_dim: torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
            "v_cache": {seq_dim: torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        },
        "input_names": ("input_ids", "position_ids"),
        "output_names": ("logits",),
        "state_names": (KEY_CACHE_NAME, VALUE_CACHE_NAME),
    }


def export_multifunction_with_kernels(model, entries, custom_kernels):
    """`export_to_coreai_multifunction` + the custom-kernel registration hook.

    Mirrors `gemma4_metal_mlp.export_to_coreai_with_kernels`, which registers the kernels with
    the converter BEFORE `add_pytorch_module` (that call validates the exported program against
    the converter's known lowerings), but loops over entrypoints so both functions land in one
    AIProgram with deduplicated weights.
    """
    model.eval()
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels(custom_kernels)
    for entrypoint_name, spec in entries:
        def export_fn(module, _inputs=spec["reference_inputs"], _dyn=spec.get("dynamic_shapes")):
            with torch.no_grad():
                ep = torch.export.export(module, args=(), kwargs=_inputs, dynamic_shapes=_dyn)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            return ep

        converter.add_pytorch_module(
            model, export_fn=export_fn, externalize_modules=None,
            input_names=spec["input_names"], output_names=spec["output_names"],
            state_names=spec["state_names"], entrypoint_name=entrypoint_name,
        )
    register_custom_torch_lowering(converter)
    return converter.to_coreai()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()

    print(f"loading BitCPM-8B from gguf (layers={args.num_layers or 'all'}) ...", flush=True)
    model, mv_kernel = load_bitcpm8b_from_gguf(GGUF, num_layers=args.num_layers, dtype=DTYPE)
    cfg = model.config

    # engine sizes the logits buffer as ceil(vocab/64)*64; BitCPM's 73448 (%64=40) aborts warm-up
    old = cfg.vocab_size
    new = ((old + 63) // 64) * 64
    if new != old:
        head = nn.Linear(cfg.hidden_size, new, bias=False).to(model.lm_head.weight.dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[:old].copy_(model.lm_head.weight)
        model.lm_head, cfg.vocab_size = head, new
        print(f"padded lm_head {old} -> {new}", flush=True)

    gemm_kernel = build_tern_gemm_kernel(args.chunk, "bitcpm_ternary_gemm")
    n = swap_linears(model, mv_kernel, gemm_kernel, args.chunk)
    print(f"dual-kernel linears: {n}", flush=True)

    name = f"bitcpm_8b_ternary_pf{args.chunk}" + (f"_l{args.num_layers}" if args.num_layers else "")
    entries = [("main", build_spec(cfg, args.max_ctx, 1)),
               ("prefill", build_spec(cfg, args.max_ctx, args.chunk))]
    print(f"exporting multifunction (main S=1 + prefill S={args.chunk}) ...", flush=True)
    prog = export_multifunction_with_kernels(model, entries, [mv_kernel, gemm_kernel])
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
    (out_dir / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": "openbmb/BitCPM-CANN-8B", "vocab_size": cfg.vocab_size,
                     "max_context_length": args.max_ctx, "embedded_tokenizer": True,
                     "function_map": {"main": ["main", "prefill"]}},
        "source": {"model_definition": "torch", "hf_model_id": "openbmb/BitCPM-CANN-8B"},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }, indent=2))
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(HF, trust_remote_code=True).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
