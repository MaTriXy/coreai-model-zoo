#!/bin/bash
# Sideload VibeVoice-Realtime-0.5B assets into the coreai-audio app data container, then run the
# on-device self-test. Run AFTER installing coreai-audio (iPhone UNLOCKED + trusted).
# Assets land in Library/Application Support/VibeVoiceAssets (VibeVoiceAssets resolves it via
# .applicationSupportDirectory). ~2.8 GB (5 AOT .aimodelc + compact device_bundle + golden.f32).
#
#   ./sideload_ios.sh <device-udid>            # udid: xcrun devicectl list devices
#   (iPhone 17 Pro this box = A6F3E849-1947-5202-9AD1-9C881CA58EEF)
#
# The .aimodelc were AOT-compiled from artifacts/*/*.aimodel with:
#   xcrun coreai-build compile <aimodel> --output artifacts_ios/ --platform iOS \
#       --architecture h18p --preferred-compute gpu --min-deployment-version 27.0
set -euo pipefail
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Users/majimadaisuke/Downloads/Xcode-beta.app/Contents/Developer}"
DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="Library/Application Support/VibeVoiceAssets"

copy() {
  echo ">> $(basename "$1")"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "$1" --destination "$DEST/$(basename "$1")"
}

# 5 AOT graph bundles (dirs)
for g in vibevoice_diffusion_head_fp16 vibevoice_connector_fp16 vibevoice_decoder_fp16_t64 \
         vibevoice_mainlm_fp16_decode_cl512 vibevoice_ttslm_fp16_decode_cl512; do
  copy "$HERE/artifacts_ios/$g.h18p.aimodelc"
done
# compact host inputs + golden (device_bundle/) — meta.json is the root marker
for f in "$HERE"/device_bundle/*; do copy "$f"; done

# CoreAIKit host assets (voices_coreai/, built by pack_voice_presets.py + the embed table):
# the Dialogue tab drives KitDialogue, which wants glue/ + voices/ + embed/ next to the graphs.
# Skipped automatically if the tree hasn't been packed yet (the self-test doesn't need them).
if [ -d "$HERE/voices_coreai" ]; then
  for sub in glue voices embed; do
    [ -d "$HERE/voices_coreai/$sub" ] && copy "$HERE/voices_coreai/$sub"
  done
fi

echo "Done. Headless gate:"
echo "  xcrun devicectl device process launch --device $DEV \\"
echo "    --environment-variables '{\"VIBEVOICE_SELFTEST\":\"1\",\"VV_RESULT\":\"/tmp/vv.txt\"}' $BID"
echo "  # then read the app console; the self-test logs '[VV] gate vs golden: cos=... -> PASS/FAIL'"
