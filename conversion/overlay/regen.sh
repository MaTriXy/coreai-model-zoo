#!/bin/zsh
# Regenerate this overlay from a live coreai-models checkout.
# Run after adding/edting model authoring code in the checkout:
#   ./regen.sh /path/to/coreai-models
set -euo pipefail

SRC="${1:?usage: regen.sh /path/to/coreai-models}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_COMMIT="$(awk -F': *' '/^commit:/{print $2}' "$HERE/BASE")"

cd "$SRC"

# Tracked edits -> one patch
git diff "$BASE_COMMIT" -- python/src/coreai_models > "$HERE/patches/python-overlay.patch"

# Untracked package files -> files/ (mirror; removes files deleted upstream)
rm -rf "$HERE/files"
mkdir -p "$HERE/files"
git status --porcelain | awk '/^\?\? python\/src\//{print $2}' | grep -v __pycache__ | \
while IFS= read -r f; do
  mkdir -p "$HERE/files/$(dirname "$f")"
  cp "$f" "$HERE/files/$f"
done

echo "patch: $(wc -l < "$HERE/patches/python-overlay.patch") lines"
echo "files: $(find "$HERE/files" -type f | wc -l | tr -d ' ')"
