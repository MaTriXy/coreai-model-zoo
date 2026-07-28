#!/usr/bin/env python3
"""Generate llms.txt — the one fetch that tells a model what this repository knows.

Why it matters here specifically: Apple's own Core AI documentation is a JavaScript
application. Fetching `developer.apple.com/documentation/coreai` returns 63 words, all of
them "This page requires JavaScript." So for an agent answering a Core AI question there is
no fetchable authoritative source, and the verified notes in `knowledge/` are among the few
that exist. They are useless if nothing announces them.

Follows the llms.txt convention: an H1, a blockquote summary, then link sections where every
entry is `[title](url): description`. Descriptions are lifted from `knowledge/README.md`, so
the index and the notes cannot disagree — this file is generated, never hand-edited.

    python3 scripts/gen_llms_txt.py           # regenerate
    python3 scripts/gen_llms_txt.py --check   # CI: fail if stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KNOWLEDGE = REPO / "knowledge" / "README.md"
OUT = REPO / "llms.txt"
RAW = "https://raw.githubusercontent.com/john-rocky/coreai-model-zoo/main"

# `- [`file.md`](file.md) — description, possibly wrapped over the following indented lines.`
ENTRY = re.compile(r"^- \[`([^`]+)`\]\(([^)]+)\)\s*[—-]\s*(.*)$")
SECTION = re.compile(r"^## +(.*)$")

PREAMBLE = f"""\
# Core AI model zoo

> Community ports of open models to Apple's Core AI runtime (`.aimodel`, iOS/macOS 27), each
> with the recipe that produced it, plus a knowledge base of verified findings about the
> runtime itself. Apple's own Core AI documentation is a JavaScript application and returns no
> readable body to a fetcher, so these notes are written to be the fetchable answer: measured,
> dated, and specific about what was verified and how.

## Start here

- [AGENTS.md]({RAW}/AGENTS.md): the porting contract in one file — why conversion is not format
  conversion, the two gates every port gets, the traps agents specifically hit, and which
  actions stay a human's call.
- [README.md]({RAW}/README.md): the catalog itself — every model, its card, its Hugging Face
  repo, and the one-line Swift call that runs it.
- [models/index.json]({RAW}/models/index.json): the same catalog machine-readable. Per recipe:
  `status` (is the configuration recorded), `source_model` (what it was converted from), and
  `gate_transcript` (is the numerical check against the original published, and where).
- [PORTING.md]({RAW}/PORTING.md): the full walk from a Hugging Face checkpoint to a verified
  bundle on an iPhone, with two worked examples.
- [SECURITY.md]({RAW}/SECURITY.md): what the integrity story is, including the parts that are
  absent — pinned revisions rather than signatures, and no checksum manifest.
"""

FOOTER = f"""
## Optional

- [CONTRIBUTING.md]({RAW}/CONTRIBUTING.md): what an accepted port must clear, and the device
  gate — the one step a contributor without an iOS 27 device can hand back.
- [BENCHMARKS.md]({RAW}/BENCHMARKS.md): community-submitted device measurements, explicitly not
  a controlled-environment benchmark.
"""


def parse_sections() -> list[tuple[str, list[tuple[str, str, str]]]]:
    """`knowledge/README.md` -> [(section title, [(file, url, description)])]."""
    sections: list[tuple[str, list[tuple[str, str, str]]]] = []
    current: list[tuple[str, str, str]] = []
    title = "Knowledge base"
    pending: list[str] | None = None
    entry: tuple[str, str] | None = None

    def flush() -> None:
        nonlocal pending, entry
        if entry and pending is not None:
            desc = " ".join(" ".join(pending).split())
            current.append((entry[0], entry[1], desc.rstrip(".")))
        pending, entry = None, None

    for line in KNOWLEDGE.read_text().splitlines():
        if m := SECTION.match(line):
            flush()
            if current:
                sections.append((title, current.copy()))
                current.clear()
            title = m.group(1).strip()
        elif m := ENTRY.match(line):
            flush()
            entry, pending = (m.group(1), m.group(2)), [m.group(3)]
        elif pending is not None and line.startswith("  ") and line.strip():
            pending.append(line.strip())
        elif not line.strip():
            flush()
    flush()
    if current:
        sections.append((title, current))
    return sections


def render() -> str:
    out = [PREAMBLE]
    for title, entries in parse_sections():
        if not entries:
            continue
        out.append(f"\n## {title}\n")
        for name, url, desc in entries:
            # Strip the emphasis markers that read as noise once flattened to one line.
            clean = re.sub(r"\*\*|`", "", desc)
            out.append(f"- [{name}]({RAW}/knowledge/{url}): {clean}")
        out.append("")
    out.append(FOOTER)
    return "\n".join(out).replace("\n\n\n", "\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit non-zero if llms.txt is stale")
    args = ap.parse_args()

    want = render()
    if args.check:
        have = OUT.read_text() if OUT.exists() else ""
        if have != want:
            sys.exit("llms.txt is stale — knowledge/README.md changed without regenerating it.\n"
                     "  Fix with: python3 scripts/gen_llms_txt.py")
        print(f"OK: llms.txt matches knowledge/README.md ({want.count('](') } links)")
        return
    OUT.write_text(want)
    print(f"wrote llms.txt ({want.count('](')} links, {len(want)} bytes)")


if __name__ == "__main__":
    main()
