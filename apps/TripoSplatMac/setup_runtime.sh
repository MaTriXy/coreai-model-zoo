#!/bin/bash
# Stage the TripoSplat Core AI runtime (backend script + converted bundles + checkpoints) into a
# stable directory the app points at, so it survives reboots / new sessions.
# ~9 GB (coreai_out ~5.5G + ckpts ~3.8G).
#
# Usage: ./setup_runtime.sh <SRC> [DEST]
#   SRC  = your TripoSplat runtime dir (produced by ../../conversion/triposplat; must contain
#          app_backend.py, triposplat.py, model.py, coreai_out/, ckpts/)
#   DEST = where to stage it (default: ~/TripoSplatRuntime — the app's default "Backend dir")
set -e

SRC="${1:?usage: ./setup_runtime.sh <SRC dir with app_backend.py, coreai_out/, ckpts/> [DEST]}"
DST="${2:-$HOME/TripoSplatRuntime}"

if [ ! -f "$SRC/app_backend.py" ]; then
  echo "Source not found: $SRC/app_backend.py" >&2
  echo "Point SRC at the triposplat runtime dir (see ../../conversion/triposplat)." >&2
  exit 1
fi

echo "Copying runtime: $SRC -> $DST"
mkdir -p "$DST"
cp "$SRC"/app_backend.py "$SRC"/triposplat.py "$SRC"/model.py "$DST"/
cp -R "$SRC"/coreai_out "$DST"/
cp -R "$SRC"/ckpts "$DST"/
[ -d "$SRC"/static ] && cp -R "$SRC"/static "$DST"/

echo "Done. Runtime at: $DST"
echo "Set the app's 'Backend dir' to: $DST  (default is ~/TripoSplatRuntime)"
du -sh "$DST"
