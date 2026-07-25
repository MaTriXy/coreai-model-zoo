import argparse
from pathlib import Path

import export_lfm2_embeds_decode as L

# Minimal export worker — invoked as a SUBPROCESS by run_lfm2_embeds_gate.py.
# The coreai_torch converter's interleave check ("interleave must have rank (1)")
# is sensitive to the calling module's shape; this pristine, docstring-free,
# export-only entry (the proven pattern) converts cleanly where a richer runner
# module does not. Keep it minimal. Prints "EXPORT_OK <path>" on success.

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["fp16", "int8lin"])
ap.add_argument("out_dir")
ap.add_argument("--max-ctx", type=int, default=4096)
args = ap.parse_args()

cfg = L.load_lfm_config()
model = L.load_backbone(cfg)
aimodel = L.export_bundle(model, cfg, args.mode, args.max_ctx, Path(args.out_dir))
print("EXPORT_OK", aimodel)
