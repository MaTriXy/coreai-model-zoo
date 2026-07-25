#!/bin/bash
# Sideload dots.tts iOS assets into the coreai-audio app data container so the "Voice ML" tab loads
# them with no network. Run AFTER installing coreai-audio on the device, iPhone UNLOCKED + trusted.
# Assets land in Library/Application Support/DotsAssets (int4 backbone + fp16 dit/patchenc/vocoder,
# AOT-compiled for iOS h18p in ship_ios/). Footprint ~4.2 GB (.aimodelc mmap'd; needs the
# increased-memory entitlement, which coreai-audio has).
#   ./sideload_ios.sh <device-udid>      (xcrun devicectl list devices)
set -euo pipefail

DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
HERE="$(cd "$(dirname "$0")" && pwd)"
SHIP="$HERE/ship_ios"
ART="$HERE/artifacts"
DEST="Library/Application Support/DotsAssets"

copy() {  # push a file/dir individually (bulk copy can false-succeed on a wired tunnel)
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

# Lean device set (~4.2 GB) — reliable load on the iPhone. Prefill + S=84 bucket bundles dropped
# (they save ~1s but add ~2 GB → memory pressure/hang on load); Swift falls back to prefill-via-decode
# (warm) + the S=164 bucket. Re-add dots_backbone_int4_prefill_t32 / dots_dit_mf_fp16_s84 only if RAM allows.
for b in dots_backbone_int4_decode_cl512 dots_backbone_int4_prefill_t32 dots_patchenc_fp16_buf256 \
         dots_dit_mf_fp16_s164 dots_vocoder_fp16_t60; do
  copy "$SHIP/$b.h18p.aimodelc"
done
copy "$ART/dots_host_glue"
copy "$ART/tokenizer"

echo "Done. Launch coreai-audio → 'Voice ML' tab → Load → Speak  (or set DOTS_SELFTEST=1 / touch ~/.dots_selftest)."
