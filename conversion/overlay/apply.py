#!/usr/bin/env python3
"""Apply the zoo's coreai_models python overlay onto a pinned apple/coreai-models checkout.

Usage:
    python3 apply.py /path/to/coreai-models [--force]

What it does:
  1. Verifies the target checkout is at the pinned base commit (see BASE).
  2. Applies patches/python-overlay.patch (edits to tracked export/registry/primitive files).
  3. Copies files/ (re-authored model definitions) into the package tree.

After applying, install the package into the venv you run conversion scripts with:
    cd /path/to/coreai-models && pip install -e python/

--force skips the clean-tree check (NOT the base-commit check). Use it only when
re-applying on top of a previous apply.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_base() -> str:
    for line in (HERE / "BASE").read_text().splitlines():
        if line.startswith("commit:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("BASE file has no commit line")


def git(target: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(target), *args], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="path to the coreai-models checkout")
    ap.add_argument("--force", action="store_true", help="skip the clean-tree check")
    args = ap.parse_args()
    target = args.target.resolve()

    base = read_base()
    head = git(target, "rev-parse", "HEAD")
    if head != base:
        raise SystemExit(
            f"target HEAD {head[:12]} != pinned base {base[:12]} — "
            "check out the pinned commit first (see BASE)"
        )

    dirty = git(target, "status", "--porcelain", "--", "python/src/coreai_models")
    if dirty and not args.force:
        raise SystemExit(
            "target python/src/coreai_models is not clean — refusing to apply "
            "(re-run with --force to apply on top anyway)"
        )

    patch = HERE / "patches" / "python-overlay.patch"
    check = subprocess.run(
        ["git", "-C", str(target), "apply", "--check", str(patch)],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        if args.force:
            print("patch --check failed (probably already applied); skipping patch step")
        else:
            raise SystemExit(f"patch does not apply cleanly:\n{check.stderr}")
    else:
        subprocess.check_call(["git", "-C", str(target), "apply", str(patch)])
        print(f"applied {patch.name}")

    files_root = HERE / "files"
    copied = 0
    for src in sorted(files_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(files_root)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"copied {copied} files")
    print("done — now: pip install -e python/ (inside your conversion venv)")


if __name__ == "__main__":
    main()
