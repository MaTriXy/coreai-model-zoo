"""Export the Z-Image S3-DiT to Core AI (fp16 or int8lin).

The graph is the NativeZDiT block stack over host-prepped inputs (patchify / RoPE /
pad-mask / unpatchify stay on the host — see zimage_host.py). Bit-exact vs diffusers
in fp32 (parity_dit_torch.py: corr 1.000000). n_cap must be a multiple of
SEQ_MULTI_OF (=32); the host pads the caption up to round_up(L, 32) and those pad
tokens are real attention context.

Ship graph (Mac): bf16 + both axes dynamic — one bundle for 256/512/1024 and any
prompt length. fp16 NaNs this model; int8 is slower than bf16 (see knowledge/zimage-port.md).

Run (coreai-models venv, from conversion/zimage/):
  python export_dit.py bf16 --size 512 --cap 32 --dyn-cap --dyn-img   # ship graph
  python export_dit.py bf16 --size 512 --cap 32 --layers 2            # fast probe
"""
import argparse
import shutil
from pathlib import Path

import torch

from zimage_dit_native import NativeZDiT

DTYPE = torch.float16


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
            # Z-Image norms are RMSNorm (1D weight) — per-block axis-1 quant would
            # fail on rank-1 tensors; keep them unquantized.
            "diffusers.models.normalization.RMSNorm": None,
        },
    }


def act_quant_config(dtype: str = "int8") -> dict:
    """Weight AND activation int8 -> a true int8 x int8 integer matmul.

    This is the recipe the device-proven LiteRT/Android port shipped ("INTEGER-int8
    graph; the weight-only-FLOAT path hangs/overflows on the GPU delegate"). Our
    linear_quant_config() is exactly that rejected weight-only path: it dequantizes
    to float, which (a) costs 2.4-2.7x speed, (b) lets fp16 overflow inside the float
    matmul, and (c) makes the AOT compiler materialize fp32 weights.
    Static activation quant -> needs calibration data (we use the real oracle inputs).
    """
    cfg = linear_quant_config(dtype)
    # Scope activation quant to Linear ONLY. Putting op_input_spec on global_config
    # quantizes the inputs of EVERY op (softmax, mul, add, ...) and NaNs from step 0.
    wspec = {"dtype": dtype, "qscheme": "symmetric_with_clipping",
             "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}
    aspec = {"dtype": dtype, "qscheme": "symmetric",
             "granularity": {"type": "per_tensor"}}
    cfg["module_type_configs"]["torch.nn.modules.linear.Linear"] = {
        "op_state_spec": {"weight": wspec},
        "op_input_spec": {"*": aspec},
        "op_output_spec": None,
    }
    return cfg


def build_calibration_fn(rm, size, dtype, order):
    """Real DiT inputs from the captured oracle (per-step latent/adaln + true caption).
    The 512 latents are centre-cropped to the probe's latent size — value distribution
    (what the activation ranges are calibrated on) is preserved."""
    import json as _json
    from zimage_host import build_native_inputs
    here = Path(__file__).parent
    ora = here / "oracle"
    meta = _json.load(open(ora / "meta.json"))
    lat, Lc = meta["lat"], meta["cap_cond_L"]

    def _load(n, shape):
        import numpy as _np
        return torch.from_numpy(_np.fromfile(ora / f"{n}.f32", "<f4")).reshape(shape).float()

    cap = _load("cap_cond", (1, Lc, 2560))[0]
    tgt = size // 8

    def fn():
        data = []
        for s in range(meta["steps"]):
            latent = _load(f"latent_{s}", (1, 16, 1, lat, lat))[0][:, :, :tgt, :tgt]
            ins = build_native_inputs(rm, latent, cap)
            ins["adaln"] = _load(f"adaln_{s}", (1, 256))
            data.append(tuple(ins[k].to(dtype) for k in order))
        return data

    return fn


def build_ref_inputs(size: int, n_cap: int, dtype):
    lat = size // 8
    n_img = (lat // 2) * (lat // 2)
    hd = 128
    cshape = (1, n_img, hd // 2)
    kshape = (1, n_cap, hd // 2)
    ref = {
        "img_tokens": torch.randn(1, n_img, 64, dtype=dtype),
        "cap_feats": torch.randn(1, n_cap, 2560, dtype=dtype),
        "adaln": torch.randn(1, 256, dtype=dtype),
        "x_cos": torch.randn(*cshape, dtype=dtype),
        "x_sin": torch.randn(*cshape, dtype=dtype),
        "cap_cos": torch.randn(*kshape, dtype=dtype),
        "cap_sin": torch.randn(*kshape, dtype=dtype),
        "x_pad_mask": torch.zeros(1, n_img, 1, dtype=dtype),
        "cap_pad_mask": torch.zeros(1, n_cap, 1, dtype=dtype),
    }
    return ref, n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "bf16", "int8lin"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cap", type=int, default=32, help="n_cap (multiple of 32)")
    ap.add_argument("--layers", type=int, default=None, help="subset of main layers (probe)")
    ap.add_argument("--base", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                    help="activation dtype for int8lin (auto=bf16). fp16 NaNs (Z-Image overflows); "
                         "fp32 is the LiteRT-proven int8-weights + FP32-compute recipe and is the "
                         "only 16-bit-free dtype iOS AOT accepts unambiguously")
    ap.add_argument("--dyn-cap", action="store_true",
                    help="make the caption axis dynamic (one graph for any prompt length; "
                         "REQUIRED for long prompts because cond/uncond have different n_cap)")
    ap.add_argument("--cap-max", type=int, default=256, help="max n_cap when --dyn-cap")
    ap.add_argument("--residual-scale", type=float, default=1.0, metavar="C",
                    help="fp16-safe rescale: divide feed_forward.w2 and attention.to_out by C "
                         "(output-exact; RMSNorm follows both). REQUIRED for fp16 graphs — those "
                         "two linears reach 3.1e5 > 65504 and NaN the DiT at sampler step 2. C=16 works.")
    ap.add_argument("--act-quant", action="store_true",
                    help="quantize ACTIVATIONS to int8 too (true int8xint8 matmul; the LiteRT/"
                         "Android recipe). Calibrated on the captured oracle inputs.")
    ap.add_argument("--io-fp32", action="store_true",
                    help="fp32 graph inputs/outputs with bf16 weights+compute — required for a "
                         "Swift host (bfloat16 NDArrays cannot be filled from Swift)")
    ap.add_argument("--update-scale", type=float, default=1.0, metavar="C",
                    help="fp16-safe: divide feed_forward.w2 / attention.to_out by C (output-exact)")
    ap.add_argument("--dyn-img", action="store_true",
                    help="make the image-token axis dynamic (one graph for 256/512/1024)")
    ap.add_argument("--img-max", type=int, default=4096, help="max n_img when --dyn-img")
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()
    assert args.cap % 32 == 0, "n_cap must be a multiple of 32"

    tag = f"L{args.layers}" if args.layers is not None else "full"
    suffix = "" if args.base == "auto" else f"_{args.base}act"
    if args.dyn_cap:
        suffix += "_dyncap"
    if args.dyn_img:
        suffix += "_dynimg"
    if args.residual_scale != 1.0 or args.update_scale != 1.0:
        suffix += f"_rs{int(args.residual_scale)}u{int(args.update_scale)}"
    if args.act_quant:
        suffix += "_actq"
    if args.io_fp32:
        suffix += "_iofp32"
    name = f"zimage_dit_{args.size}_cap{args.cap}_{tag}_native_{args.mode}{suffix}"

    from coreai_models.export.macos import export_to_coreai
    import coreai.runtime as rt

    print(f"[dit] loading transformer (fp32) ...", flush=True)
    from diffusers import ZImageTransformer2DModel
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    # bf16 base for bf16 AND int8lin by default (int8 weights on bf16 activations keep
    # the residual-overflow safety Z-Image needs; fp16 activations NaN on late steps).
    # --base fp16 forces fp16 activations (faster int8 kernel? numerics may NaN).
    if args.base != "auto":
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.base]
    else:
        dtype = torch.bfloat16 if args.mode in ("bf16", "int8lin") else DTYPE
    deploy = NativeZDiT(rm, n_layers=args.layers,
                        residual_scale=args.residual_scale,
                        update_scale=args.update_scale,
                        io_fp32=args.io_fp32).eval().to(dtype)

    ref, n_img = build_ref_inputs(args.size, args.cap,
                                 torch.float32 if args.io_fp32 else dtype)
    dyn = {k: None for k in ref}  # fully static
    axis = 1
    if args.dyn_cap:
        from torch.export import Dim
        ncap = Dim("ncap", min=32, max=args.cap_max)
        dyn["cap_feats"] = {1: ncap}
        dyn["cap_pad_mask"] = {1: ncap}
        dyn["cap_cos"] = {axis: ncap}
        dyn["cap_sin"] = {axis: ncap}
    if args.dyn_img:
        from torch.export import Dim
        nimg = Dim("nimg", min=256, max=args.img_max)
        dyn["img_tokens"] = {1: nimg}
        dyn["x_pad_mask"] = {1: nimg}
        dyn["x_cos"] = {axis: nimg}
        dyn["x_sin"] = {axis: nimg}
    print(f"[dit] graph: n_img={n_img} n_cap={args.cap}{'(dyn)' if args.dyn_cap else ''} "
          f"layers={tag} mode={args.mode}", flush=True)

    model = deploy
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        if args.act_quant:
            print("[dit] quantizing (int8 weights + int8 ACTIVATIONS, calibrated) ...", flush=True)
            calib = build_calibration_fn(rm, args.size, dtype, tuple(ref.keys()))
            model = quantize_pytorch_model(
                deploy, tuple(ref.values()), dyn, act_quant_config("int8"),
                calibration_data_fn=calib)
        else:
            print("[dit] quantizing (int8lin, weight-only) ...", flush=True)
            model = quantize_pytorch_model(
                deploy, tuple(ref.values()), dyn, linear_quant_config("int8"))

    print("[dit] exporting to Core AI ...", flush=True)
    prog = export_to_coreai(
        model, ref, dynamic_shapes=dyn,
        input_names=tuple(ref.keys()), output_names=("velocity",))
    print("[dit] optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[dit] saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"[dit] bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
