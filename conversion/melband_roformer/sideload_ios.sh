#!/bin/bash
# Sideload Mel-Band RoFormer iOS assets into the coreai-audio app's data container, so the
# "Separate" tab loads them without a network download. Run AFTER installing coreai-audio on the
# device (Xcode), with the iPhone UNLOCKED and trusted. Assets land in
# Library/Application Support/SeparateAssets, which the app resolves via .applicationSupportDirectory.
#
#   ./sideload_ios.sh <device-udid>
#   # device-udid: `xcrun devicectl list devices`  (or Xcode > Devices)
#
# The AOT bundle comes from: coreai-build compile artifacts/mbr_full_fp16/mbr_full_fp16.aimodel \
#   --output artifacts/ios_full_h18p --platform iOS --architecture h18p --preferred-compute gpu
# (already staged in ship_ios/). golden_*.f32 (demo clip + self-test target) come from ship_macos/.
set -euo pipefail

DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="Library/Application Support/SeparateAssets"

copy() {  # copy a file/dir into the container's SeparateAssets dir
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

copy "$HERE/ship_ios/mbr_full_fp16.h18p.aimodelc"     # the AOT bundle (STFT + RoFormer + iSTFT)
copy "$HERE/ship_ios/metadata.json"
copy "$HERE/ship_macos/golden_raw.f32"                # 8 s demo clip + self-test input
copy "$HERE/ship_macos/golden_vocals.f32"             # self-test target (SEPARATE_SELFTEST=1)

echo "Done. Launch coreai-audio → Separate tab → Load → Demo clip (or choose a song)."
echo "Headless verify:  devicectl device process launch ... --environment-variables '{\"SEPARATE_SELFTEST\":\"1\"}'"
