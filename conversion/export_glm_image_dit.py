"""Export the GLM-Image DiT (GlmImageTransformer2DModel) to Core AI (T2I).

The T2I DiT is a stateless flow-matching MMDiT (30 layers, inner 4096, in/out 16ch,
patch 2) with prior-token conditioning and NO kv-cache (that path is edit-only).
Wrapper: pre-computed rope (rope depends only on H,W) + float `prior_scale`
(1=cond, 0=uncond) replacing the export-hostile boolean prior drop. Wrapper is
byte-exact vs diffusers (coreai/_glmimg_dit_parity.py).

Graph contract (bespoke image-pipeline loop):
  (hidden_states [1,16,h/8,w/8], encoder_hidden_states [1,T,1472] (T dyn),
   prior_token_id [1,N] int32, prior_scale [1,1,1], timestep [1],
   target_size [1,2], crop_coords [1,2], cos [N,128], sin [N,128]) -> noise [1,16,h/8,w/8]

Run (coreai-models venv, from conversion/):
  python export_glm_image_dit.py [int8lin|fp16] --tf-dir <snap>/transformer --size 512
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

DTYPE = torch.float16


class GlmImageDiTWrapper(nn.Module):
    """GlmImageTransformer2DModel.forward for export: precomputed rope + float
    prior_scale (no boolean index), no kv-cache."""

    def __init__(self, tf):
        super().__init__()
        self.tf = tf
        self.p = tf.config.patch_size

    def forward(self, hidden_states, encoder_hidden_states, prior_token_id,
                prior_scale, timestep, target_size, crop_coords, cos, sin):
        tf = self.tf
        b, c, h, w = hidden_states.shape
        ph, pw = h // self.p, w // self.p
        rope = (cos, sin)
        hs = tf.image_projector(hidden_states)
        ehs = tf.glyph_projector(encoder_hidden_states)
        prior = tf.prior_projector(tf.prior_token_embedding(prior_token_id) * prior_scale)
        hs = hs + prior
        temb = tf.time_condition_embed(timestep, target_size, crop_coords, hs.dtype)
        for block in tf.transformer_blocks:
            hs, ehs = block(hs, ehs, temb, rope, None, None, kv_cache=None)
        hs = tf.norm_out(hs, temb)
        hs = tf.proj_out(hs)
        hs = hs.reshape(b, ph, pw, -1, self.p, self.p)
        return hs.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)


def linear_quant_config(dtype: str = "int8") -> dict:
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
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.normalization.LayerNorm": None,
        },
    }


def build_spec(wrap, size: int, dtype, hd: int = 128, text_dim: int = 1472,
               text_len: int = 1):
    # STATIC text length: the diffusers attn processor does
    # `hidden_states.split([text_len, image_len])`; a dynamic text_len makes the
    # split sizes dynamic Nodes that the Core AI converter rejects. Glyph-free
    # prompts always yield text_len=1 (empty glyph embed) for both cond+uncond,
    # so a static T=1 graph covers all glyph-free T2I. (Glyph text-in-image =
    # follow-up needing a padded+masked or per-length variant.)
    lat = size // 8
    n_patch = (lat // 2) * (lat // 2)
    ref = {
        "hidden_states": torch.randn(1, 16, lat, lat, dtype=dtype),
        "encoder_hidden_states": torch.randn(1, text_len, text_dim, dtype=dtype),
        "prior_token_id": torch.randint(0, 16384, (1, n_patch), dtype=torch.int32),
        "prior_scale": torch.ones(1, 1, 1, dtype=dtype),
        "timestep": torch.tensor([500.0], dtype=dtype),
        "target_size": torch.tensor([[size, size]], dtype=dtype),
        "crop_coords": torch.zeros(1, 2, dtype=dtype),
        "cos": torch.zeros(n_patch, hd, dtype=dtype),
        "sin": torch.zeros(n_patch, hd, dtype=dtype),
    }
    dyn = {k: None for k in ref}  # fully static
    return {
        "reference_inputs": ref, "dynamic_shapes": dyn,
        "input_names": tuple(ref.keys()), "output_names": ("noise",),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "int8lin"])
    ap.add_argument("--tf-dir", required=True)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    name = f"glm_image_dit_{args.size}_{args.mode}"
    print(f"loading DiT transformer ({args.mode}) ...", flush=True)
    from diffusers import GlmImageTransformer2DModel
    tf = GlmImageTransformer2DModel.from_pretrained(args.tf_dir, torch_dtype=DTYPE).eval()
    wrap = GlmImageDiTWrapper(tf).eval()
    hd = tf.config.attention_head_dim

    spec = build_spec(wrap, args.size, DTYPE, hd=hd, text_dim=tf.config.text_embed_dim)

    model = wrap
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("quantizing (int8lin) ...", flush=True)
        model = quantize_pytorch_model(
            wrap, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], linear_quant_config("int8"))

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    import coreai.runtime as rt

    print(f"exporting DiT graph ({name}) ...", flush=True)
    prog = export_to_coreai(
        model, spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"], output_names=spec["output_names"],
        state_names=None, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
