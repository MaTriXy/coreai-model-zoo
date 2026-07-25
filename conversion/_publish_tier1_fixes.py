#!/usr/bin/env python3
"""One-off: fix the tier-1 defects `zoo_verify.py` found in already-published bundles.

Three changes, all recorded in models/_INVENTORY.md before they were made:

  1. 10 Gemma 4 E2B/E4B bundles ship no chat template while their source repo does, so a
     host cannot format prompts. Adds `tokenizer/chat_template.jinja`, fetched from the
     source repo each bundle names in its own metadata.json. Purely additive.
  2. 12 Gemma 4 E2B/E4B bundles set `eos_token = "<eos>"`, which is the end of a
     *sequence*, not the end of a *turn* — a host stopping on it runs past the reply.
     Gemma 4 ends turns with `<turn|>` (`eot_token` upstream), which is what the zoo's
     own 12B/31B bundles already ship and what apps/CoreAIChat hardcodes as a workaround
     (`Gemma4VLBackend.EOT = 106`). Sets `eos_token = "<turn|>"` to match.
  3. The MinerU layout decoder's metadata names a local absolute path as its
     `hf_model_id` and tokenizer — published, and wrong. Points it at the upstream model
     its sibling bundle names.

Run with --dry-run first. Every file this touches was backed up to
`$ZOO_WORK_ROOT/_hf_backup/2026-07-25-chat-template-eos/` with the pre-change repo
revisions; Hugging Face keeps the full history besides, so each commit is revertible.

    <venv>/bin/python conversion/_publish_tier1_fixes.py --dry-run
    <venv>/bin/python conversion/_publish_tier1_fixes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hf_catalog import Catalog, bundle_paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EOT = "<turn|>"
MINERU = "mlboydaisuke/MinerU2.5-Pro-CoreAI"
MINERU_SOURCE = "opendatalab/MinerU2.5-Pro-2605-1.2B"

COMMIT_TEMPLATE = (
    "Add the chat template and stop at the end of a turn\n\n"
    "- tokenizer/chat_template.jinja: copied from {source} — the export script dropped it, "
    "so a host had no way to format prompts.\n"
    "- eos_token: <eos> -> <turn|>. Gemma 4 ends a turn with <turn|>; <eos> ends the "
    "sequence, so a host stopping on it ran past the reply. Matches the Gemma-4-12B/31B "
    "bundles in this catalog.\n\n"
    "Found by conversion/zoo_verify.py in github.com/john-rocky/coreai-model-zoo."
)
MINERU_COMMIT = (
    "metadata: name the upstream model instead of a local path\n\n"
    "source.hf_model_id and language.tokenizer held an absolute path from the machine that "
    "did the export. Points them at the model the sibling decoder bundle names.\n\n"
    "Found by conversion/zoo_verify.py in github.com/john-rocky/coreai-model-zoo."
)


def plan(cat: Catalog) -> dict[str, list[dict]]:
    """(repo -> operations), derived from the recorded verification results."""
    verify = json.loads((REPO / "models" / "_VERIFY.json").read_text())["bundles"]
    ops: dict[str, list[dict]] = defaultdict(list)

    for b in verify:
        repo, bundle = b["repo"], b["bundle"]
        needs_template = any(c["check"] == "chat template" and c["status"] == "FAIL"
                             for c in b["checks"])
        needs_eos = any(c["check"] == "eos vs eot" for c in b["checks"])
        if not (needs_template or needs_eos):
            continue
        files = [s["rfilename"] for s in (cat.repo(repo) or {}).get("siblings", [])]
        paths = bundle_paths(bundle, files)
        meta = cat.json_file(repo, paths["metadata"]) or {}
        source = (meta.get("source") or {}).get("hf_model_id")

        if needs_template:
            template = cat.file(source, "chat_template.jinja") if source else None
            if not template:
                print(f"  !! {repo}/{bundle}: no chat_template.jinja at {source}", file=sys.stderr)
            else:
                ops[repo].append({
                    "path": f"{bundle}/tokenizer/chat_template.jinja",
                    "content": template,
                    "what": f"add chat template ({len(template)} B from {source})",
                    "source": source,
                })
        if needs_eos:
            rel = paths["tokenizer_config"]
            raw = cat.file(repo, rel) or ""
            tok = json.loads(raw) if raw else {}
            before = tok.get("eos_token")
            if before == EOT:
                continue
            # Surgical: these files are already 2-space JSON, so replacing the one line
            # keeps the published diff to one line instead of a whole-file reformat.
            old_line = f'  "eos_token": {json.dumps(before)},'
            if raw.count(old_line) != 1:
                print(f"  !! {repo}/{bundle}: eos_token line not unique, skipping",
                      file=sys.stderr)
                continue
            ops[repo].append({
                "path": rel,
                "content": raw.replace(old_line, f'  "eos_token": {json.dumps(EOT)},'),
                "what": f"eos_token {before!r} -> {EOT!r}",
                "source": source,
            })

    meta = cat.json_file(MINERU, "layout/decoder/metadata.json") or {}
    if str((meta.get("source") or {}).get("hf_model_id", "")).startswith("/"):
        meta["source"]["hf_model_id"] = MINERU_SOURCE
        if str(meta.get("language", {}).get("tokenizer", "")).startswith("/"):
            meta["language"]["tokenizer"] = "tokenizer"
        ops[MINERU].append({
            "path": "layout/decoder/metadata.json",
            "content": json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            "what": f"hf_model_id -> {MINERU_SOURCE}",
            "source": MINERU_SOURCE,
        })
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="restrict to one repo id")
    args = ap.parse_args()

    cat = Catalog()
    ops = plan(cat)
    if args.only:
        ops = {k: v for k, v in ops.items() if k == args.only}

    total = sum(len(v) for v in ops.values())
    for repo, changes in ops.items():
        print(f"\n{repo}  ({len(changes)} file operations)")
        for c in changes:
            print(f"  {c['what']:<52} {c['path']}")
    print(f"\n{total} file operations across {len(ops)} repos")
    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    for repo, changes in ops.items():
        message = MINERU_COMMIT if repo == MINERU else COMMIT_TEMPLATE.format(
            source=next((c["source"] for c in changes if c["source"]), "the source repo"))
        commit = api.create_commit(
            repo_id=repo,
            operations=[CommitOperationAdd(path_in_repo=c["path"],
                                           path_or_fileobj=c["content"].encode())
                        for c in changes],
            commit_message=message.splitlines()[0],
            commit_description="\n".join(message.splitlines()[1:]).strip(),
        )
        print(f"{repo}: committed {len(changes)} files -> {commit.oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
