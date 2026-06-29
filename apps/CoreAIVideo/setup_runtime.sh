#!/bin/bash
# Stage the CoreAIVideo runtime into a stable directory the app points at.
# The scratch working dir is session-temporary; this survives reboots / new sessions.
#
# Stages (no T5 weights — the bundle does that compute, so the 19 GB PixArt encoder is NOT copied):
#   ltx_video/                                   (the LTX-Video package, host glue)
#   coreai_out/{dit_fp16,vae_fp16,t5_bf16}.aimodel   (~13.5 GB, the converted nets)
#   ckpts/ltxv-2b-0.9.6-distilled-04-25.safetensors  (~5.9 GB, transformer + vae torch weights)
#   ckpts/pixart/tokenizer/                      (~0.8 MB, T5 tokenizer only)
# Total ~19.5 GB.
#
# Usage: ./setup_runtime.sh <scratch LTX-Video dir> [dest]
set -e

SRC="${1:?pass the scratch LTX-Video dir (holds ltx_video/, coreai_out/, ckpts/)}"
DST="${2:-$HOME/CoreAIVideoRuntime}"

for need in ltx_video coreai_out/dit_fp16.aimodel coreai_out/vae_fp16.aimodel \
            coreai_out/t5_bf16.aimodel ckpts/ltxv-2b-0.9.6-distilled-04-25.safetensors \
            ckpts/pixart/tokenizer; do
  if [ ! -e "$SRC/$need" ]; then
    echo "Missing $SRC/$need — run the conversion scripts first (conversion/ltxvideo/)." >&2
    exit 1
  fi
done

echo "Staging runtime: $SRC -> $DST"
mkdir -p "$DST/coreai_out" "$DST/ckpts/pixart"
cp -R "$SRC/ltx_video" "$DST/"
cp -R "$SRC"/coreai_out/{dit_fp16,vae_fp16,t5_bf16}.aimodel "$DST/coreai_out/"
cp "$SRC"/ckpts/ltxv-2b-0.9.6-distilled-04-25.safetensors "$DST/ckpts/"
cp -R "$SRC/ckpts/pixart/tokenizer" "$DST/ckpts/pixart/"

echo "Done. Runtime at: $DST"
echo "Point the app's 'Runtime' field at: $DST  (default is ~/CoreAIVideoRuntime)"
du -sh "$DST"
