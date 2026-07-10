"""Export the Z-Image text encoder (Qwen3-4B) to Core AI.

Graph: inputs_embeds [1,L,2560] + additive 4D mask [1,1,L,L] (causal AND
non-padding) -> Qwen3 stack -> PENULTIMATE hidden hidden_states[-2] [1,L,2560].
Host does: chat-template tokenize -> right-pad to L -> embed_tokens gather ->
inputs_embeds, and builds the mask; position_ids = arange(L) baked in. The
padding keys are masked, so valid-token outputs are pad-length independent
(verified corr 1.000000 vs the pipeline's cap embeds). bf16 (fp16 risk); int8lin
for ship size.

Run (coreai-models venv, from conversion/zimage/):
  python export_encoder.py bf16 --L 64                 # full 36-layer
  python export_encoder.py bf16 --L 64 --layers 2      # convertibility probe
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype, "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.normalization.LayerNorm": None,
            "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": None,
        },
    }


class EncWrap(nn.Module):
    def __init__(self, te, L):
        super().__init__()
        self.te = te
        self.register_buffer("pos", torch.arange(L)[None])

    def forward(self, inputs_embeds, mask):
        out = self.te(inputs_embeds=inputs_embeds, attention_mask=mask,
                      position_ids=self.pos, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "bf16", "int8lin"])
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.mode in ("bf16", "int8lin") else DTYPE

    tag = f"L{args.layers}" if args.layers is not None else "full"
    name = f"zimage_encoder_seq{args.L}_{tag}_{args.mode}"

    from coreai_models.export.macos import export_to_coreai
    import coreai.runtime as rt

    print("[enc] loading Qwen3 text_encoder ...", flush=True)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    te = Qwen3Model.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="text_encoder",
        torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    if args.layers is not None:
        te.layers = te.layers[:args.layers]
    wrap = EncWrap(te, args.L).eval().to(dtype)

    neg = torch.finfo(dtype).min
    ref = {
        "inputs_embeds": torch.randn(1, args.L, 2560, dtype=dtype),
        "mask": torch.triu(torch.full((1, 1, args.L, args.L), neg, dtype=dtype), 1),
    }
    dyn = {k: None for k in ref}
    print(f"[enc] graph: L={args.L} layers={tag} mode={args.mode}", flush=True)

    model = wrap
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("[enc] quantizing (int8lin) ...", flush=True)
        model = quantize_pytorch_model(
            wrap, tuple(ref.values()), dyn, linear_quant_config("int8"))

    print("[enc] exporting to Core AI ...", flush=True)
    prog = export_to_coreai(model, ref, dynamic_shapes=dyn,
                            input_names=tuple(ref.keys()), output_names=("penultimate",))
    print("[enc] optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[enc] saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"[enc] bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
