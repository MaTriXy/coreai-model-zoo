#!/usr/bin/env python3
"""Aggregate community bench blobs into BENCHMARKS.md (device x model median table).

Sources
  * GitHub issues labeled `bench-result` on john-rocky/coreai-model-zoo (default) —
    the public audit log. Each issue body carries one JSON blob produced by the
    CoreAIChat Bench tab (see .github/ISSUE_TEMPLATE/bench-result.yml).
  * --local DIR: *.json blob files (the on-device DoD loop / offline testing).

Trust model (mirrors the app side): the harness measured, the submitter only pasted.
Blobs are validated hard — schema_version, kind, and an EXACT protocol match
(pb-random-v1: 128-token seed-0 prompt, 256 greedy tokens, chunk_threshold 1,
1 cold + 3 warm). Anything else is rejected loudly, never silently dropped.

Field-data policy: sloppy environments show up as outliers, so we post-filter on
environment metadata instead of trusting the numbers: Low Power Mode ON or a
serious/critical thermal state BEFORE the run excludes a blob from the medians
(still counted + listed as excluded). Cells are the median across submissions of
each submission's own median warm decode tok/s; n >= 3 shows plain, n < 3 is
marked provisional.

Usage
  python3 scripts/aggregate_bench.py                 # fetch issues, write BENCHMARKS.md
  python3 scripts/aggregate_bench.py --local DIR     # also read local blobs
  python3 scripts/aggregate_bench.py --local-only DIR
  GH_TOKEN=... raises the API rate limit (optional).
"""

import argparse
import json
import os
import re
import statistics
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = "john-rocky/coreai-model-zoo"
SCHEMA_VERSION = 1
BLOB_KIND = "coreai-community-bench"
PROTOCOL = {
    "name": "pb-random-v1",
    "prompt_tokens": 128,
    "max_tokens": 256,
    "prompt_seed": 0,
    "temperature": 0,
    "chunk_threshold": 1,
    "cold_runs": 1,
    "warm_runs": 3,
}
# Canonical column order; unknown model ids append after these.
MODEL_ORDER = ["qwen3.5-0.8b", "lfm2.5-1.2b", "granite-4.0-h-1b"]

# utsname.machine -> (marketing name, chip). Unknown ids show raw — extend as needed.
DEVICE_NAMES = {
    "iPhone13,1": ("iPhone 12 mini", "A14"),
    "iPhone13,2": ("iPhone 12", "A14"),
    "iPhone13,3": ("iPhone 12 Pro", "A14"),
    "iPhone13,4": ("iPhone 12 Pro Max", "A14"),
    "iPhone14,4": ("iPhone 13 mini", "A15"),
    "iPhone14,5": ("iPhone 13", "A15"),
    "iPhone14,2": ("iPhone 13 Pro", "A15"),
    "iPhone14,3": ("iPhone 13 Pro Max", "A15"),
    "iPhone14,6": ("iPhone SE (3rd gen)", "A15"),
    "iPhone14,7": ("iPhone 14", "A15"),
    "iPhone14,8": ("iPhone 14 Plus", "A15"),
    "iPhone15,2": ("iPhone 14 Pro", "A16"),
    "iPhone15,3": ("iPhone 14 Pro Max", "A16"),
    "iPhone15,4": ("iPhone 15", "A16"),
    "iPhone15,5": ("iPhone 15 Plus", "A16"),
    "iPhone16,1": ("iPhone 15 Pro", "A17 Pro"),
    "iPhone16,2": ("iPhone 15 Pro Max", "A17 Pro"),
    "iPhone17,1": ("iPhone 16 Pro", "A18 Pro"),
    "iPhone17,2": ("iPhone 16 Pro Max", "A18 Pro"),
    "iPhone17,3": ("iPhone 16", "A18"),
    "iPhone17,4": ("iPhone 16 Plus", "A18"),
    "iPhone17,5": ("iPhone 16e", "A18"),
    "iPhone18,1": ("iPhone 17 Pro", "A19 Pro"),
    "iPhone18,2": ("iPhone 17 Pro Max", "A19 Pro"),
    "iPhone18,3": ("iPhone 17", "A19"),
    "iPhone18,4": ("iPhone Air", "A19 Pro"),
}


def log(msg):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- blob intake

def extract_blob_text(issue_body):
    """The issue form (render: json) fences the paste; accept a bare object too."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", issue_body, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", issue_body, re.DOTALL)
    return m.group(0) if m else None


def validate(blob):
    """Return a rejection reason, or None when the blob is acceptable."""
    if not isinstance(blob, dict):
        return "not a JSON object"
    if blob.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version {blob.get('schema_version')!r} != {SCHEMA_VERSION}"
    if blob.get("kind") != BLOB_KIND:
        return f"kind {blob.get('kind')!r}"
    proto = blob.get("protocol") or {}
    for key, want in PROTOCOL.items():
        if proto.get(key) != want:
            return f"protocol.{key} {proto.get(key)!r} != {want!r} (protocol is fixed)"
    for path in (("device", "model_identifier"), ("model", "id"),
                 ("environment", "low_power_mode"), ("results", "runs")):
        node = blob
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return f"missing {'.'.join(path)}"
    warm = [r for r in blob["results"]["runs"]
            if r.get("kind") == "warm" and isinstance(r.get("decode_tok_s"), (int, float))]
    if len(warm) != PROTOCOL["warm_runs"]:
        return f"expected {PROTOCOL['warm_runs']} warm runs, got {len(warm)}"
    if any(r["decode_tok_s"] <= 0 for r in warm):
        return "non-positive warm decode_tok_s"
    return None


def excluded_reason(blob):
    """Environment post-filter (field-data policy). None = counts toward medians."""
    env = blob["environment"]
    if env.get("low_power_mode"):
        return "Low Power Mode on"
    if env.get("thermal_state_before") in ("serious", "critical"):
        return f"thermal {env['thermal_state_before']} before run"
    return None


class Submission:
    def __init__(self, blob, source, submitter):
        self.blob = blob
        self.source = source          # issue URL or file path
        self.submitter = submitter    # GitHub login or "local"
        self.device = blob["device"]["model_identifier"]
        self.model = blob["model"]["id"]
        warm = [r["decode_tok_s"] for r in blob["results"]["runs"] if r["kind"] == "warm"]
        self.warm_decode_median = statistics.median(warm)
        prefill = [r["prefill_tok_s"] for r in blob["results"]["runs"]
                   if r["kind"] == "warm" and isinstance(r.get("prefill_tok_s"), (int, float))]
        self.warm_prefill_median = statistics.median(prefill) if prefill else None
        self.excluded = excluded_reason(blob)


# ---------------------------------------------------------------- sources

def fetch_issues(repo):
    """All open+closed issues labeled bench-result (closing an issue does NOT
    remove the row — the issue stream is an append-only audit log)."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    issues, page = [], 1
    while True:
        url = (f"https://api.github.com/repos/{repo}/issues"
               f"?labels=bench-result&state=all&per_page=100&page={page}")
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "aggregate-bench",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        issues += [i for i in batch if "pull_request" not in i]
        if len(batch) < 100:
            return issues
        page += 1


def load_submissions(repo, local_dirs, use_github):
    subs, rejected = [], []
    if use_github:
        try:
            issues = fetch_issues(repo)
            log(f"fetched {len(issues)} bench-result issue(s) from {repo}")
        except Exception as e:  # noqa: BLE001 — network failure is a normal offline case
            log(f"WARNING: could not fetch issues from {repo}: {e}")
            issues = []
        for issue in issues:
            ref = f"#{issue['number']}"
            text = extract_blob_text(issue.get("body") or "")
            if not text:
                rejected.append((ref, "no JSON blob found in body"))
                continue
            try:
                blob = json.loads(text)
            except json.JSONDecodeError as e:
                rejected.append((ref, f"invalid JSON: {e}"))
                continue
            reason = validate(blob)
            if reason:
                rejected.append((ref, reason))
                continue
            subs.append(Submission(blob, issue["html_url"],
                                   (issue.get("user") or {}).get("login", "unknown")))
    for d in local_dirs:
        for path in sorted(Path(d).glob("*.json")):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                rejected.append((str(path), f"unreadable: {e}"))
                continue
            reason = validate(blob)
            if reason:
                rejected.append((str(path), reason))
                continue
            subs.append(Submission(blob, str(path), "local"))
    return subs, rejected


# ---------------------------------------------------------------- table

def device_label(identifier):
    if identifier in DEVICE_NAMES:
        name, chip = DEVICE_NAMES[identifier]
        return f"{name} ({chip}, `{identifier}`)"
    return f"`{identifier}`"


def device_sort_key(identifier):
    m = re.match(r"([A-Za-z]+)(\d+),(\d+)", identifier)
    if m:
        return (m.group(1), -int(m.group(2)), -int(m.group(3)))
    return (identifier, 0, 0)


def build_markdown(subs, rejected, repo):
    included = [s for s in subs if not s.excluded]
    excluded = [s for s in subs if s.excluded]
    models = MODEL_ORDER + sorted({s.model for s in subs} - set(MODEL_ORDER))
    devices = sorted({s.device for s in included}, key=device_sort_key)

    cells = {}  # (device, model) -> (median, n)
    for s in included:
        cells.setdefault((s.device, s.model), []).append(s.warm_decode_median)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Community Benchmarks",
        "",
        "**Community field data** — measured by the Bench tab of",
        "[CoreAIChat](apps/CoreAIChat) ([TestFlight](https://testflight.apple.com/join/bK4P7xby))",
        "on contributors' own devices and submitted as",
        "[bench-result issues](https://github.com/" + repo + "/issues?q=label%3Abench-result)",
        "(the public audit log). The app measures and builds the result blob; no number in",
        "this table was typed by a human. This is NOT a controlled-environment benchmark —",
        "background load and heat show up here as real-world variance.",
        "",
        "**Protocol** `pb-random-v1`: fixed 128-token random prompt (seed 0) → 256 greedy",
        "decode tokens, S=1 prefill (`COREAI_CHUNK_THRESHOLD=1`), 1 cold + 3 warm runs on a",
        "freshly created engine. Cell = median across submissions of each submission's",
        "median **warm decode tok/s**; `n` = accepted submissions. Cells with n < 3 are",
        "**provisional** (marked \\*). Blobs with Low Power Mode on or a serious/critical",
        "thermal state before the run are excluded from medians (counted below).",
        "",
        "**Add your device**: TestFlight → Bench tab → Run → *Submit on GitHub*. Your",
        "device becomes a row here on the next aggregation",
        "(`python3 scripts/aggregate_bench.py`).",
        "",
        f"_Generated by `scripts/aggregate_bench.py` — do not edit by hand. Last run: {now}._",
        "",
        "## Decode tok/s (median warm)",
        "",
    ]

    header = "| Device | " + " | ".join(models) + " |"
    sep = "|---" * (len(models) + 1) + "|"
    lines += [header, sep]
    for dev in devices:
        row = [device_label(dev)]
        for model in models:
            vals = cells.get((dev, model))
            if not vals:
                row.append("—")
            else:
                med = statistics.median(vals)
                mark = "" if len(vals) >= 3 else "\\*"
                row.append(f"{med:.1f}{mark} (n={len(vals)})")
        lines.append("| " + " | ".join(row) + " |")
    if not devices:
        lines.append("| _no accepted submissions yet_ |" + " — |" * len(models))

    lines += [
        "",
        f"Accepted submissions: **{len(included)}** · excluded by environment filter:"
        f" **{len(excluded)}** · rejected (schema/protocol): **{len(rejected)}**",
        "",
    ]

    if excluded:
        lines += ["<details><summary>Excluded submissions (environment filter)</summary>", ""]
        for s in excluded:
            lines.append(f"- {s.device} · {s.model} — {s.excluded} ({s.source})")
        lines += ["", "</details>", ""]
    if rejected:
        lines += ["<details><summary>Rejected submissions (schema/protocol)</summary>", ""]
        for ref, reason in rejected:
            lines.append(f"- {ref} — {reason}")
        lines += ["", "</details>", ""]

    contributors = sorted({s.submitter for s in included if s.submitter not in ("local", "unknown")})
    if contributors:
        lines += ["## Contributors", "",
                  " ".join(f"[@{c}](https://github.com/{c})" for c in contributors), ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT, help=f"GitHub repo (default {REPO_DEFAULT})")
    ap.add_argument("--local", action="append", default=[],
                    help="directory of local *.json blobs (repeatable), in addition to issues")
    ap.add_argument("--local-only", action="append", default=[],
                    help="directory of local *.json blobs; skip the GitHub fetch")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "BENCHMARKS.md"),
                    help="output path (default: repo-root BENCHMARKS.md)")
    args = ap.parse_args()

    local_dirs = args.local + args.local_only
    subs, rejected = load_submissions(args.repo, local_dirs, use_github=not args.local_only)
    log(f"accepted {len([s for s in subs if not s.excluded])}, "
        f"excluded {len([s for s in subs if s.excluded])}, rejected {len(rejected)}")
    for ref, reason in rejected:
        log(f"  rejected {ref}: {reason}")

    md = build_markdown(subs, rejected, args.repo)
    Path(args.out).write_text(md)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
