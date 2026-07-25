import argparse
from pathlib import Path

import torch

import detokenizer as DT
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

# Minimal detok-backbone export worker (run as a SUBPROCESS; keeps the converter in a
# clean module). Static-shape masked prefill: inputs_embeds [1,S,512] -> spec [1,S,1282].
# Prints "EXPORT_OK <path>".

ap = argparse.ArgumentParser()
ap.add_argument("seq", type=int)                       # S = 6 * num_frames
ap.add_argument("out_dir")
ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
ap.add_argument("--window", type=int, default=None,
                help="override sliding-window size (0 = full causal; for lowering isolation)")
ap.add_argument("--fp32-attn", action="store_true",
                help="keep attention q/k/v/out projections fp32 (Milestone-A precision lesson)")
args = ap.parse_args()

if args.window is not None:
    DT.WINDOW = args.window                             # Attn.__init__ reads the module global
dtype = torch.float16 if args.dtype == "float16" else torch.float32
detok = DT.load_detokenizer(dtype)
if args.fp32_attn and dtype == torch.float16:
    import torch.nn as _nn
    for layer in detok.backbone.layers:
        if hasattr(layer, "self_attn"):
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.out_proj):
                lin.weight = _nn.Parameter(lin.weight.float())
spec = DT.DetokSpec(detok).eval()

example = {"inputs_embeds": torch.randn(1, args.seq, DT.DIM, dtype=dtype)}
prog = export_to_coreai(
    spec, example, dynamic_shapes=None,
    input_names=("inputs_embeds",), output_names=("spec",),
    state_names=None, externalize_modules=list(_EXTERNALIZE_SPECS))
prog.optimize()

import coreai.runtime as rt  # noqa: E402  (lazy, after converter)

out = Path(args.out_dir)
if out.exists():
    import shutil
    shutil.rmtree(out)
out.mkdir(parents=True)
aim = out / f"{out.name}.aimodel"
meta = rt.AIModelAssetMetadata()
meta.license = "lfm-open-license-v1.0"
prog.save_asset(aim, meta)
sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file()) / 1e6
print(f"[save] {aim} ({sz:.1f} MB)")
print("EXPORT_OK", aim)
