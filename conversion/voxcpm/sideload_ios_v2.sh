#!/bin/bash
# Sideload VoxCPM2 (2B) iOS model assets into the coreai-audio app's data container, so the "Voice 2B"
# tab loads them without a network download. Run AFTER installing coreai-audio on the device (Xcode),
# with the iPhone UNLOCKED and trusted. Assets land in Library/Application Support/VoxCPMAssets
# (shared with the v1 "Voice" tab; v2 files are voxcpm2_* / voxcpm2_host_glue / tokenizer2).
#
#   ./sideload_ios_v2.sh <device-udid>            # full: int8 decode + int8 prefill + fp16 diffusion/VAE
#   MINIMAL=1 ./sideload_ios_v2.sh <device-udid>  # skip prefill bundles (-1.56 GB; app falls back to
#                                                 #   prefill-via-decode, bit-exact, slightly higher TTFB)
#
# Bundles are int8 base/res + fp16 feat_decoder/encoder/vocoder, AOT-compiled for iOS h18p:
#   coreai-build compile <b>.aimodel --output artifacts/ios_v2 --platform iOS --architecture h18p
#     --preferred-compute gpu --min-deployment-version 27.0
# Footprint: full = 4.9 GB, minimal = 3.3 GB (.aimodelc are mmap'd; needs the increased-memory entitlement).
set -euo pipefail

DEV="${1:?usage: sideload_ios_v2.sh <device-udid>   (xcrun devicectl list devices)}"
BID="com.daisukemajima.coreaiaudio"
ART="$(cd "$(dirname "$0")" && pwd)/artifacts"
DEST="Library/Application Support/VoxCPMAssets"

copy() {  # copy a file/dir into the container's VoxCPMAssets dir (push individually — bulk can false-succeed)
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

BUNDLES=(voxcpm2_base_int8_decode_cl512 voxcpm2_res_int8_decode_cl512
         voxcpm2_feat_decoder_fp16 voxcpm2_feat_encoder_fp16 voxcpm2_vocoder_fp16_t8)
if [ -z "${MINIMAL:-}" ]; then
  BUNDLES+=(voxcpm2_base_int8_prefill_t32 voxcpm2_res_int8_prefill_t32)
fi

for b in "${BUNDLES[@]}"; do
  copy "$ART/ios_v2/$b.h18p.aimodelc"
done
copy "$ART/voxcpm2_host_glue"
copy "$ART/tokenizer2"

echo "Done. Launch coreai-audio → 'Voice 2B' tab → Load (int8) → Speak  (or set VOXCPM2_SELFTEST=1)."
