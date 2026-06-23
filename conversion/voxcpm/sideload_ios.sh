#!/bin/bash
# Sideload VoxCPM iOS model assets into the coreai-audio app's data container, so the "Voice" tab
# loads them without a network download. Run AFTER installing coreai-audio on the device (Xcode),
# with the iPhone UNLOCKED and trusted. Assets land in Library/Application Support/VoxCPMAssets,
# which the app resolves via FileManager .applicationSupportDirectory.
#
#   ./sideload_ios.sh <device-udid>
#   # device-udid: `xcrun devicectl list devices`  (or Xcode > Devices)
#
# The 5 AOT bundles come from: coreai-build compile ... --platform iOS --architecture h18p
#   (see this dir's README; outputs in artifacts/ios/). Tokenizer + host glue come from artifacts/.
set -euo pipefail

DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
ART="$(cd "$(dirname "$0")" && pwd)/artifacts"
DEST="Library/Application Support/VoxCPMAssets"

copy() {  # copy a file/dir into the container's VoxCPMAssets dir
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

for b in voxcpm_base_int8_decode_cl512 voxcpm_res_int8_decode_cl512 \
         voxcpm_feat_decoder_fp16 voxcpm_feat_encoder_fp16 voxcpm_vocoder_fp16_t12; do
  copy "$ART/ios/$b.h18p.aimodelc"
done
copy "$ART/voxcpm_host_glue"
copy "$ART/tokenizer"

echo "Done. Launch coreai-audio → Voice tab → Load → Speak (or set VOXCPM_SELFTEST=1 in the scheme)."
