#!/usr/bin/env bash
# make-testflight.sh — archive CoreAIChat and upload the build to TestFlight.
#
# One-shot pipeline: patch the coreai-models clone -> bundle the Gemma tokenizer ->
# xcodegen -> archive -> export -> upload via the App Store Connect API.
#
# Prerequisites (the parts this script can NOT do for you):
#   1. An App Store Connect API key (Users and Access > Integrations > App Store Connect API),
#      role "App Manager" or higher. You get a .p8 file, a Key ID, and an Issuer ID.
#   2. An app record in App Store Connect for bundle id com.daisukemajima.CoreAIChat
#      (App Store Connect > Apps > +). Automatic signing registers the App ID itself.
#   3. `huggingface-cli login` done, with the Gemma license accepted on
#      https://huggingface.co/google/gemma-4-E2B-it (needed only for the bundled Gemma tokenizer).
#
# Usage:
#   export ASC_KEY_P8=/path/to/AuthKey_XXXXXXXXXX.p8
#   export ASC_KEY_ID=XXXXXXXXXX
#   export ASC_ISSUER_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#   export DEVELOPER_DIR=/Users/$USER/Downloads/Xcode-beta.app/Contents/Developer
#   ./make-testflight.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"           # apps/CoreAIChat
COMMUNITY_ROOT="$(cd "$APP_DIR/../.." && pwd)"      # coreai-models-community
COREAI_MODELS="$COMMUNITY_ROOT/coreai-models"       # the patched Apple clone (symlink ok)
BUILD="$APP_DIR/build"
ARCHIVE="$BUILD/CoreAIChat.xcarchive"

: "${DEVELOPER_DIR:?set DEVELOPER_DIR to the Xcode 27 beta Developer dir}"
: "${ASC_KEY_P8:?set ASC_KEY_P8 to the App Store Connect API .p8 path}"
: "${ASC_KEY_ID:?set ASC_KEY_ID}"
: "${ASC_ISSUER_ID:?set ASC_ISSUER_ID}"

echo "==> Xcode: $(xcodebuild -version | head -1)"

# 1. Patch stack on the coreai-models clone (idempotent). Apply only what is missing. This clone is
#    a live working repo (the model ports live here), so NEVER reset it. A patch that neither applies
#    nor reverses usually means the equivalent engine change is already hand-integrated — warn and
#    continue; a genuinely missing change surfaces as a compile error below, not a lost working tree.
echo "==> Verifying coreai-models patch stack"
for p in coreai-shared-product coreai-pipelined-extra-states \
         coreai-pipelined-per-token-inputs coreai-pipelined-static-inputs; do
  patch="$COMMUNITY_ROOT/apps/$p.patch"
  if git -C "$COREAI_MODELS" apply --reverse --check "$patch" 2>/dev/null; then
    echo "    already applied: $p"
  elif git -C "$COREAI_MODELS" apply --check "$patch" 2>/dev/null; then
    git -C "$COREAI_MODELS" apply "$patch"; echo "    applied:         $p"
  else
    echo "    WARN: $p does not apply cleanly — assuming it is already integrated in this clone."
  fi
done

# 2. Bundle the Gemma tokenizer.json (the only piece the in-app download does NOT bring — pipelined
#    Qwen/LFM/Granite carry their own tokenizer inside the downloaded bundle; Gemma reads Resources).
TOK="$APP_DIR/Resources/tokenizer/tokenizer.json"
if [ ! -f "$TOK" ]; then
  echo "==> Fetching Gemma tokenizer.json into Resources/tokenizer/"
  huggingface-cli download google/gemma-4-E2B-it tokenizer.json \
    --local-dir "$APP_DIR/Resources/tokenizer"
fi

# 3. Generate the project from project.yml.
echo "==> xcodegen generate"
( cd "$APP_DIR" && xcodegen generate )

# 4. Archive (Release, generic iOS device, automatic signing via the API key).
echo "==> Archiving"
xcodebuild archive \
  -project "$APP_DIR/CoreAIChat.xcodeproj" \
  -scheme CoreAIChat \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE" \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$ASC_KEY_P8" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID"

# 4b. Inject the top-level CFBundleIconName. actool writes CFBundleIcons:CFBundlePrimaryIcon
#     into the generated Info.plist, but NOT the top-level CFBundleIconName key that App Store
#     Connect validation requires (error 90713) — the GENERATE_INFOPLIST_FILE flow on Xcode 27
#     beta does not add it, and INFOPLIST_KEY_CFBundleIconName is ignored. Patch the archived app
#     before export; -exportArchive re-signs, sealing the change. Idempotent.
APP="$ARCHIVE/Products/Applications/CoreAIChat.app"
/usr/libexec/PlistBuddy -c "Add :CFBundleIconName string AppIcon" "$APP/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Set :CFBundleIconName AppIcon" "$APP/Info.plist"

# 5. Export the .ipa locally (ExportOptions.plist uses destination=export — no upload yet).
echo "==> Exporting .ipa"
xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$APP_DIR/ExportOptions.plist" \
  -exportPath "$BUILD/export" \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$ASC_KEY_P8" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID"

# 6. Validate, then upload to TestFlight. altool reads the key from ~/.appstoreconnect/private_keys.
#    cp exits 1 when source and destination are the same file (key already in place) — don't let
#    that abort the upload under set -e.
mkdir -p ~/.appstoreconnect/private_keys
cp -f "$ASC_KEY_P8" ~/.appstoreconnect/private_keys/ 2>/dev/null || true
echo "==> Validating"
xcrun altool --validate-app -f "$BUILD/export/CoreAIChat.ipa" -t ios \
  --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID"
echo "==> Uploading to TestFlight"
xcrun altool --upload-app -f "$BUILD/export/CoreAIChat.ipa" -t ios \
  --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID"

echo "==> Done. Watch processing at https://appstoreconnect.apple.com (TestFlight tab)."
echo "    Beta-SDK builds (Xcode 27 + iOS 27 beta SDK) are accepted for TestFlight internal and"
echo "    external testing per the ASC release notes (2026-06-08 / 06-10)."
