#!/bin/bash
# Sideload the Nemotron 3.5 ASR streaming bundle into the coreai-audio app's data container
# (Documents/Models/Nemotron — where TranscribeModel/NemotronSelfTest look before the Hub).
# Pushes FILE BY FILE with copy-back verification (the wired CoreDevice tunnel can return
# exit 0 on a dropped transfer; --remove-existing-content is FORBIDDEN — it wipes the container).
#
#   ./sideload_ios.sh <device-udid>

set -uo pipefail

DEV="${1:?usage: sideload_ios.sh <device-udid>}"
BID="com.daisukemajima.coreaiaudio"
ART="$(cd "$(dirname "$0")" && pwd)/artifacts"
DEST="Documents/Models/Nemotron"

push() {  # push one file to a container-relative destination, retrying until copy-back verifies
  # EVERY file is verified by a full pull + md5 — `copy to` returns exit 0 on dropped transfers
  # (a 2.4 GB push once truncated at 490 MB with "success"); pulls run ~32 MB/s so this is cheap.
  local src="$1" dst="$2" tries=0
  local size want got
  size=$(stat -f%z "$src")
  want=$(md5 -q "$src")
  while :; do
    tries=$((tries + 1))
    xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
      --domain-identifier "$BID" --source "$src" --destination "$dst" >/dev/null 2>&1
    local tmp; tmp=$(mktemp -d)/check
    xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
      --domain-identifier "$BID" --source "$dst" --destination "$tmp" >/dev/null 2>&1
    got=$(md5 -q "$tmp" 2>/dev/null)
    rm -f "$tmp"
    if [ "$got" = "$want" ]; then
      echo ">> $dst ($((size / 1000000)) MB, attempt $tries) ✓ md5"
      return 0
    fi
    [ $tries -ge 5 ] && { echo "!! $dst FAILED after $tries attempts"; return 1; }
    sleep 2
  done
}

# JIT graphs (small) + tokenizer, then the two AOT conformer halves (whole-dir pushes; the
# single 24-layer AOT bundle fails to LOAD on-device, hence the a/b split).
for m in nemotron_asr_stream_pre_first_float16.aimodel \
         nemotron_asr_stream_pre_float16.aimodel \
         nemotron_asr_predict_float32.aimodel \
         nemotron_asr_joint_float32.aimodel; do
  for f in metadata.json main.mlirb main.hash; do
    push "$ART/$m/$f" "$DEST/$m/$f" || exit 1
  done
done
push "$ART/bundle_assets/tokenizer.json" "$DEST/tokenizer.json" || exit 1
push "$ART/bundle_assets/tokenizer_config.json" "$DEST/tokenizer_config.json" || exit 1

for half in a b; do
  AOT="ios/nemotron_asr_stream_conformer_${half}_float16.h18p.aimodelc"
  AOTDST="$DEST/nemotron_asr_stream_conformer_${half}_float16.h18p.aimodelc"
  (cd "$ART" && find "$AOT" -type f) | while read -r rel; do
    push "$ART/$rel" "$AOTDST/${rel#"$AOT"/}" || exit 1
  done
done

echo "Done. Push /tmp/libri1.wav to Documents/ and launch with NEMOTRON_SELFTEST=1."
