#!/usr/bin/env python3
"""Tier-1 verification: does a published bundle agree with the model it came from?

Four checks, none of which needs an oracle, a device, or a single byte of
weights — so this runs over the whole published catalog in minutes:

    eos / bos       the bundle's tokenizer_config vs the source repo's
    chat template   the bundle's template vs the source repo's, byte for byte
    context         the bundle's max_context_length vs the source's positions
    dtype           the precision the bundle name claims vs the one it declares

    python3 conversion/zoo_verify.py mlboydaisuke/Gemma-4-12B-CoreAI
    python3 conversion/zoo_verify.py --all --json models/_VERIFY.json
    python3 conversion/zoo_verify.py --local exports/my_bundle --source Qwen/Qwen3.5-0.8B

Expected values come from the **source repository**, resolved from the bundle's
own `metadata.json` (`source.hf_model_id`, else `language.tokenizer`). Nothing is
transcribed by hand, so nothing goes stale.

A port may deviate from its source on purpose — swapping `eos_token` for the
end-of-turn token is a real, defensible ship-time edit. Deviations are therefore
reported as DIFF, not FAIL, until `models/<name>/verify.toml` records the
expected value and the reason; after that the recorded value is the expectation
and an unexplained deviation fails.

Statuses: `ok`, `DIFF` (deviates from the source, undeclared), `FAIL` (wrong on
its own terms), `info`, `skipped`. A check that cannot run reports `skipped` —
never `ok`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hf_catalog import Catalog, bundles_of, bundle_paths, repo_format  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUTHORS = ["mlboydaisuke"]
# Escape hatch for a published repo no recipe names; contributor-hosted ports come from
# their recipes (contributor_repos), so `--all` covers a contributed port the day its PR
# lands instead of the day someone remembers to add it here.
EXTRA_REPOS: list[str] = []


def contributor_repos() -> list[str]:
    """`hf_repo` values in models/*/recipe.toml that belong to someone else's account."""
    ours = {a.lower() for a in AUTHORS}
    named: set[str] = set()
    for path in sorted((REPO / "models").glob("*/recipe.toml")):
        with open(path, "rb") as fh:
            for entry in tomllib.load(fh).values():
                if repo := entry.get("hf_repo"):
                    named.add(repo)
    return sorted(r for r in named if r.split("/")[0].lower() not in ours)

PRECISION = re.compile(
    r"(?<![a-z0-9])(int2|int4km|int4lin(?:sym)?|int4hu|int4|int8hu|int8lin|int8sym|int8|"
    r"mixed48|mixedbit|sym8|fp4|fp8|fp16|float16|bf16|bfloat16|float32|fp32)(?![a-z0-9])"
)
QUANTIZED = ("int", "sym", "fp4", "fp8", "mixed")

OK, DIFF, FAIL, INFO, SKIP = "ok", "DIFF", "FAIL", "info", "skipped"


# --------------------------------------------------------------------------
# expectations


def family_of(repo_id: str) -> str | None:
    """models/<family>/ that claims this Hugging Face repo, via its recipe."""
    for path in sorted((REPO / "models").glob("*/recipe.toml")):
        with open(path, "rb") as fh:
            for recipe in tomllib.load(fh).values():
                if recipe.get("hf_repo") == repo_id:
                    return path.parent.name
    return None


def load_expected(repo_id: str) -> dict:
    """Declared expectations for a repo's bundles, if anyone wrote them down.

    `models/<family>/verify.toml`, `[config]` section — beside the card and the
    recipe. Present only where a port deviates from its source on purpose (or
    where the source cannot be reached).
    """
    family = family_of(repo_id)
    path = REPO / "models" / (family or "") / "verify.toml"
    if not family or not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh).get("config", {})


def source_of(meta: dict) -> str | None:
    src = (meta.get("source") or {}).get("hf_model_id")
    return src or (meta.get("language") or {}).get("tokenizer")


def positions_of(config: dict) -> int | None:
    for cfg in (config, config.get("text_config") or {}, config.get("llm_config") or {}):
        if isinstance(cfg, dict) and cfg.get("max_position_embeddings"):
            return cfg["max_position_embeddings"]
    return None


def template_of(cat: Catalog, repo: str, tok_cfg: dict | None, jinja_path: str | None) -> str | None:
    """A chat template lives either beside tokenizer_config.json or inside it.

    The standalone `chat_template.jinja` wins, because that is the precedence
    transformers uses — comparing the field when the consumer reads the file
    reports drift that nobody will ever experience.
    """
    if jinja_path and (text := cat.file(repo, jinja_path)) is not None:
        return text
    if tok_cfg and isinstance(tok_cfg.get("chat_template"), str):
        return tok_cfg["chat_template"]
    return None


def norm(token) -> str | None:
    """tokenizer_config tokens are either a string or an AddedToken dict."""
    if token is None:
        return None
    if isinstance(token, dict):
        return token.get("content")
    return str(token)


# --------------------------------------------------------------------------
# the checks


def verify_bundle(cat: Catalog, repo: str, bundle: str, files: list[str],
                  expected: dict) -> list[dict]:
    """Run the four tier-1 checks against one published bundle."""
    paths = bundle_paths(bundle, files)
    meta = cat.json_file(repo, paths["metadata"]) if paths["metadata"] else None
    rows: list[dict] = []

    def add(check: str, status: str, detail: str) -> None:
        rows.append({"repo": repo, "bundle": bundle, "check": check,
                     "status": status, "detail": detail})

    if meta is None:
        add("metadata", FAIL, "no metadata.json — the runtime cannot identify this bundle")
        return rows

    is_llm = "language" in meta
    src = expected.get("source") or source_of(meta)

    # ---- tokenizer-side checks ------------------------------------------
    tok_path = paths["tokenizer_config"]
    if not tok_path:
        if is_llm:
            add("eos/bos", FAIL, "declares kind=llm but ships no tokenizer_config.json")
        else:
            add("eos/bos", SKIP, "not a tokenizer-bearing bundle")
            add("chat template", SKIP, "not a tokenizer-bearing bundle")
    else:
        tok = cat.json_file(repo, tok_path) or {}
        up_tok = cat.json_file(src, "tokenizer_config.json") if src else None
        if up_tok is None and not expected:
            add("eos/bos", SKIP,
                f"source tokenizer_config unavailable ({src or 'no source declared'})")
        else:
            up_tok = up_tok or {}
            for key in ("eos_token", "bos_token"):
                got, want = norm(tok.get(key)), expected.get(key, norm(up_tok.get(key)))
                label = key.split("_")[0]
                if want is None:
                    add(label, SKIP, f"source declares no {key}; bundle has {got!r}")
                elif got == want:
                    add(label, OK, f"{got!r}")
                elif key in expected:
                    add(label, FAIL, f"{got!r}, declared expectation {want!r}")
                else:
                    add(label, DIFF, f"{got!r}, source says {want!r}")
            # A chat model that stops on <eos> never stops at the end of a turn.
            eot = norm(up_tok.get("eot_token")) or norm(tok.get("eot_token"))
            if eot and norm(tok.get("eos_token")) != eot:
                add("eos vs eot", INFO,
                    f"eos {norm(tok.get('eos_token'))!r} is not the turn terminator {eot!r} — "
                    "a host that stops on eos will run past the end of the turn")

        # ---- chat template ----------------------------------------------
        got_t = template_of(cat, repo, tok, paths["chat_template"])
        up_files = [s["rfilename"] for s in (cat.repo(src) or {}).get("siblings", [])] if src else []
        up_t = template_of(cat, src, up_tok if tok_path else None,
                           "chat_template.jinja" if "chat_template.jinja" in up_files else None) if src else None
        if reason := expected.get("chat_template"):
            add("chat template", OK if got_t else FAIL,
                reason if got_t else f"declared ({reason}) but the bundle ships none")
        elif up_t is None and got_t is None:
            add("chat template", SKIP, "neither the bundle nor its source ships one")
        elif up_t is None:
            add("chat template", INFO, f"bundle ships one ({len(got_t)} B); source has none")
        elif got_t is None:
            add("chat template", FAIL,
                "source ships a chat template, the bundle ships none — the host cannot format prompts")
        elif got_t == up_t:
            add("chat template", OK, f"identical to source ({len(got_t)} B)")
        else:
            add("chat template", DIFF,
                f"differs from source ({len(got_t)} B vs {len(up_t)} B)")

    # ---- context ---------------------------------------------------------
    ctx = (meta.get("language") or {}).get("max_context_length")
    up_cfg = cat.json_file(src, "config.json") if src else None
    up_ctx = positions_of(up_cfg or {})
    if ctx is None:
        add("context", SKIP if not is_llm else FAIL,
            "no max_context_length in metadata" if is_llm else "not a context-bearing bundle")
    elif expected.get("context") is not None:
        add("context", OK if ctx == expected["context"] else FAIL,
            f"{ctx} (declared {expected['context']})")
    elif up_ctx is None:
        add("context", SKIP, f"{ctx}; source positions unknown")
    elif ctx > up_ctx:
        add("context", FAIL, f"{ctx} exceeds the source's {up_ctx} positions")
    else:
        add("context", OK, f"{ctx} of the source's {up_ctx}")

    # ---- dtype ------------------------------------------------------------
    tags = PRECISION.findall(bundle.rsplit("/", 1)[-1].lower())
    declared = meta.get("compression")
    if not tags:
        add("dtype", SKIP, "bundle name declares no precision")
    elif declared in (None, "null"):
        quant = any(t.startswith(QUANTIZED) for t in tags)
        add("dtype", INFO if quant else OK,
            f"name says {'/'.join(dict.fromkeys(tags))}; metadata declares no compression"
            if quant else f"{'/'.join(dict.fromkeys(tags))}")
    else:
        text = json.dumps(declared).lower()
        add("dtype", OK if any(t in text for t in tags) else DIFF,
            f"name says {'/'.join(dict.fromkeys(tags))}; metadata declares {text[:60]}")
    return rows


# --------------------------------------------------------------------------
# drivers


def targets(cat: Catalog, names: list[str], include_official: bool) -> list[tuple[str, list[str]]]:
    if names:
        repos = [cat.repo(n) for n in names]
    else:
        repos = [m for a in AUTHORS for m in cat.repos_by_author(a)]
        repos += [cat.repo(r) for r in dict.fromkeys(contributor_repos() + EXTRA_REPOS)]
    out = []
    for m in repos:
        if not m:
            continue
        files = [s["rfilename"] for s in m.get("siblings", [])]
        if repo_format(files) != "coreai":
            continue
        if not include_official and m["id"].endswith("-CoreAI-official") and not names:
            continue
        out.append((m["id"], files))
    return out


def print_report(rows: list[dict]) -> None:
    by_bundle: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_bundle.setdefault((r["repo"], r["bundle"]), []).append(r)
    repo = None
    for (rid, bundle), checks in by_bundle.items():
        if rid != repo:
            print(f"\n{rid}")
            repo = rid
        worst = verdict(checks)
        print(f"  {bundle}")
        for c in checks:
            if c["status"] == OK:
                continue
            print(f"    {c['check']:<14} {c['status']:<8} {c['detail']}")
        print(f"    {'VERDICT':<14} {worst}")


def verdict(checks: list[dict]) -> str:
    states = {c["status"] for c in checks}
    if FAIL in states:
        return "FAIL"
    if DIFF in states:
        return "DIFF"
    if states <= {SKIP}:
        return "SKIPPED"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", help="Hugging Face repo ids (default: the whole catalog)")
    ap.add_argument("--all", action="store_true", help="every published Core AI repo")
    ap.add_argument("--include-official", action="store_true",
                    help="also check the bench exports of Apple's own recipes")
    ap.add_argument("--bundle", help="restrict to bundles whose path contains this substring")
    ap.add_argument("--json", help="write the full result set here")
    ap.add_argument("--offline", action="store_true", help="use only the local cache")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    if not args.repos and not args.all:
        ap.error("pass one or more repo ids, or --all")

    cat = Catalog(offline=args.offline)
    rows: list[dict] = []
    for rid, files in targets(cat, args.repos, args.include_official):
        for bundle in bundles_of(files):
            if args.bundle and args.bundle not in bundle:
                continue
            rows.extend(verify_bundle(cat, rid, bundle, files, load_expected(rid)))

    if not args.quiet:
        print_report(rows)

    bundles: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        bundles.setdefault((r["repo"], r["bundle"]), []).append(r)
    tally = {v: 0 for v in ("PASS", "DIFF", "FAIL", "SKIPPED")}
    for checks in bundles.values():
        tally[verdict(checks)] += 1
    print(f"\n{len(bundles)} bundles: " + "  ".join(f"{k} {v}" for k, v in tally.items()))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"bundles": [{"repo": k[0], "bundle": k[1], "verdict": verdict(v), "checks": v}
                         for k, v in bundles.items()]}, separators=(",", ":")))
        print(f"wrote {args.json}")
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
