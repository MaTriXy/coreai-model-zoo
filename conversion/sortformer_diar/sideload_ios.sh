#!/bin/bash
# Sideload Streaming Sortformer diarize assets into the coreai-audio app's data container so the
# Transcribe tab's "Diarize" toggle works offline. Run AFTER installing coreai-audio on the device
# (Xcode / devicectl), iPhone UNLOCKED + trusted. Assets land in
# Library/Application Support/DiarizeAssets, which DiarizeAssets resolves via .applicationSupportDirectory.
#
#   ./sideload_ios.sh <device-udid>     # udid: `xcrun devicectl list devices`
#
# The AOT bundle in ship_ios/ was built from ship_macos/sortformer_float16.aimodel with:
#   coreai-build compile ship_macos/sortformer_float16.aimodel --output artifacts/ios_h18p \
#     --platform iOS --architecture h18p --preferred-compute gpu     # fixed shapes -> no --expect-frequent-reshapes
# The .f32 golden + demo wav come from ship_macos/ (platform-independent).
set -euo pipefail

DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="Library/Application Support/DiarizeAssets"

copy() {  # copy a file/dir into the container's DiarizeAssets dir
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

copy "$HERE/ship_ios/sortformer_float16.h18p.aimodelc"       # AOT graph (fixed-buffer forward_for_export)
copy "$HERE/ship_ios/metadata.json"                          # root marker + streaming params
copy "$HERE/ship_macos/sortformer_mel_filters_128x257.f32"   # librosa-slaney mel filterbank
copy "$HERE/ship_macos/golden_mel_128xT.f32"                 # self-test: demo golden mel
copy "$HERE/ship_macos/golden_total_preds.f32"               # self-test: demo golden preds
copy "$HERE/ship_macos/golden_long_mel_128xT.f32"            # self-test: long (compress) golden mel
copy "$HERE/ship_macos/golden_long_total_preds.f32"          # self-test: long golden preds
copy "$HERE/ship_macos/test_multispk_16k.wav"                # demo clip (self-test input + in-app demo)

echo "Done. Launch coreai-audio → Transcribe → Load an ASR model → toggle Diarize → Demo/Choose."
echo "Headless verify:  devicectl device process launch --device $DEV --environment-variables '{\"DIARIZE_SELFTEST\":\"1\"}' $BID"
