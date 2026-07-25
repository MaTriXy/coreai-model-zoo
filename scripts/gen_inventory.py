#!/usr/bin/env python3
"""Generate models/_INVENTORY.md — one row per published Hugging Face repo.

The catalog's worklist. For every repo the zoo publishes it records what the repo
actually contains (bundle count, format), whether this repository carries a card
and a machine-readable recipe for it, and how often it is downloaded — so later
work can be ordered by reach instead of by guess.

    python3 scripts/gen_inventory.py            # refresh models/_INVENTORY.md
    python3 scripts/gen_inventory.py --print    # dry run, print to stdout
    python3 scripts/gen_inventory.py --offline  # use only what is already cached

Reads the listing API and nothing else — no weights, no per-file fetches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "conversion"))
from _hf_catalog import Catalog, bundles_of, repo_format  # noqa: E402

AUTHORS = ["mlboydaisuke"]
# Ports published under a contributor's own account (zoo PR #6 and successors) —
# the zoo links them, so they belong in the inventory even though we do not own them.
EXTRA_REPOS = ["ukint-vs/Nanbeige4.2-3B-CoreAI"]

HF_LINK = re.compile(r"https://huggingface\.co/([\w.-]+/[\w.-]+)")
CARD_LINK = re.compile(r"\(([\w./-]*zoo/[\w.-]+\.md)\)")


def readme_mapping() -> dict[str, set[str]]:
    """HF repo id -> zoo cards, from the README model table (the published index)."""
    out: dict[str, set[str]] = defaultdict(set)
    for line in (REPO / "README.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        repos = {r for r in HF_LINK.findall(line) if not r.startswith("john-rocky/")}
        cards = {Path(c).name for c in CARD_LINK.findall(line)}
        for r in repos:
            out[r] |= cards
    return out


def card_mapping() -> dict[str, set[str]]:
    """HF repo id -> zoo cards, from links inside the cards themselves."""
    out: dict[str, set[str]] = defaultdict(set)
    for card in sorted((REPO / "zoo").glob("*.md")):
        if card.name == "README.md":  # the zoo index, not a model card
            continue
        for rid in set(HF_LINK.findall(card.read_text(errors="ignore"))):
            if not rid.startswith("john-rocky/"):
                out[rid].add(card.name)
    return out


def all_recipes() -> dict:
    with open(REPO / "conversion" / "recipes.toml", "rb") as fh:
        return tomllib.load(fh)


def recipes_by_card() -> dict[str, list[str]]:
    with open(REPO / "conversion" / "recipes.toml", "rb") as fh:
        recipes = tomllib.load(fh)
    out: dict[str, list[str]] = defaultdict(list)
    for name, r in recipes.items():
        if card := r.get("card"):
            out[Path(card).name].append(name)
    return out


def verify_results() -> dict[str, list[dict]]:
    """HF repo id -> per-bundle tier-1 verdicts, from `conversion/zoo_verify.py --json`."""
    path = REPO / "models" / "_VERIFY.json"
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = defaultdict(list)
    for b in json.loads(path.read_text())["bundles"]:
        out[b["repo"]].append(b)
    return out


def kit_slugs_by_card() -> dict[str, str]:
    path = REPO / "scripts" / "gen-cards" / "cards.json"
    if not path.exists():
        return {}
    models = json.loads(path.read_text()).get("models", {})
    return {Path(m["zooCard"]).name: slug for slug, m in models.items() if m.get("zooCard")}


def collect(cat: Catalog) -> list[dict]:
    repos = [m for a in AUTHORS for m in cat.repos_by_author(a)]
    repos += [m for m in (cat.repo(r) for r in EXTRA_REPOS) if m]

    from_readme, from_cards = readme_mapping(), card_mapping()
    by_recipe, by_kit, verified = recipes_by_card(), kit_slugs_by_card(), verify_results()

    rows = []
    for m in repos:
        rid = m["id"]
        files = [s["rfilename"] for s in m.get("siblings", [])]
        cards = sorted(from_readme.get(rid, set()) | from_cards.get(rid, set()))
        rows.append({
            "id": rid,
            "dl30": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "updated": (m.get("lastModified") or "")[:10],
            "format": repo_format(files),
            # Apple's own export recipes re-run for the bench matrix (ZOO_BLUEPRINT P2),
            # not zoo ports — they answer to Apple's repo, not to a zoo card.
            "role": "official" if rid.endswith("-CoreAI-official") else "port",
            "bundles": bundles_of(files),
            "listed": rid in from_readme,
            "cards": cards,
            "recipes": sorted({r for c in cards for r in by_recipe.get(c, [])}),
            "kit": sorted({by_kit[c] for c in cards if c in by_kit}),
            "tier1": verified.get(rid, []),
        })
    rows.sort(key=lambda r: (-r["dl30"], r["id"].lower()))
    return rows


def cell(text: str) -> str:
    """Special tokens such as `<|im_end|>` contain the column separator."""
    return text.replace("|", "\\|")


def tier1_cell(row: dict) -> str:
    """Compact per-repo tier-1 result: failures first, because that is the point."""
    if not row["tier1"]:
        return "—"
    tally: dict[str, int] = defaultdict(int)
    for b in row["tier1"]:
        tally[b["verdict"]] += 1
    parts = [f"**{tally[v]} {v}**" if v in ("FAIL", "DIFF") else f"{tally[v]} {v.lower()}"
             for v in ("FAIL", "DIFF", "PASS", "SKIPPED") if tally.get(v)]
    return " ".join(parts)


def render(rows: list[dict]) -> str:
    coreai = [r for r in rows if r["format"] == "coreai"]
    carded = [r for r in coreai if r["cards"]]
    recipe_rows = [r for r in rows if r["recipes"]]
    no_card = [r for r in coreai if not r["cards"] and r["role"] == "port"]
    official_no_card = [r for r in coreai if not r["cards"] and r["role"] == "official"]
    ambiguous = [r for r in carded if len(r["bundles"]) > 1 and not r["recipes"]]
    single = [r for r in carded if len(r["bundles"]) == 1 and not r["recipes"]]

    L = [
        "# Published model inventory",
        "",
        f"Generated by `scripts/gen_inventory.py` on {date.today().isoformat()} from the",
        "Hugging Face listing API. **Do not hand-edit** — rerun the script.",
        "",
        "One row per published repo. `bundles` counts the directories holding a",
        "`metadata.json` beside an `.aimodel` — what the runtime loads, and what",
        "verification runs against. More than one bundle means the repo ships variants,",
        "so the card and the recipe have to say which one is *the* published",
        "configuration; a single bundle answers that question by itself.",
        "",
        "`fmt` is derived from the files, not the name: `coreai` (`.aimodel`),",
        "`coreml` (pre-Core-AI ports, kept for their download history), `litert`",
        "(the Google LiteRT collaboration), `other`.",
        "",
        "| metric | count |",
        "| --- | --- |",
        f"| published repos | {len(rows)} |",
        f"| Core AI repos | {len(coreai)} |",
        f"| Core AI bundles inside them | {sum(len(r['bundles']) for r in coreai)} |",
        f"| Core AI repos with a card in `zoo/` | {len(carded)} |",
        f"| repos with a recipe in `conversion/recipes.toml` | {len(recipe_rows)} |",
        f"| Core AI repos with 0 downloads in the last 30 days | {sum(1 for r in coreai if not r['dl30'])} |",
        "",
        "## All repos, by 30-day downloads",
        "",
        "| repo | 30d DL | ♥ | fmt | role | bundles | tier-1 | card | recipe | kit |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for r in rows:
        card = ", ".join(f"[{c[:-3]}](../zoo/{c})" for c in r["cards"]) or "—"
        L.append(
            f"| [{r['id']}](https://huggingface.co/{r['id']}) | {r['dl30']} | {r['likes']} | "
            f"{r['format']} | {r['role']} | {len(r['bundles'])} | {tier1_cell(r)} | {card} | "
            f"{', '.join(f'`{x}`' for x in r['recipes']) or '—'} | "
            f"{', '.join(f'`{x}`' for x in r['kit']) or '—'} |"
        )

    defects = [(r, b) for r in rows for b in r["tier1"] if b["verdict"] in ("FAIL", "DIFF")]
    checked = sum(len(r["tier1"]) for r in rows)
    L += [
        "",
        "## Tier-1 defects",
        "",
        f"From `conversion/zoo_verify.py --all` over {checked} published bundles: the",
        "bundle's own tokenizer, chat template, context length and declared precision",
        "compared against the source repository it names in its `metadata.json`. No",
        "oracle, no device, no weights.",
        "",
        "**FAIL** = wrong on its own terms. **DIFF** = deviates from the source with no",
        "recorded reason; record the expectation in `models/<name>/verify.toml` and it",
        "becomes the bar instead of the deviation.",
        "",
    ]
    if defects:
        L += ["| repo | bundle | verdict | what |", "| --- | --- | --- | --- |"]
        for r, b in sorted(defects, key=lambda x: (x[1]["verdict"], -x[0]["dl30"])):
            for c in b["checks"]:
                if c["status"] in ("FAIL", "DIFF"):
                    L.append(f"| {r['id'].split('/')[-1]} | `{cell(b['bundle'])}` | "
                             f"{b['verdict']} | {c['check']}: {cell(c['detail'])} |")
    else:
        L.append("- (none — run `conversion/zoo_verify.py --all --json models/_VERIFY.json` first)")

    L += [
        "",
        "## Needs owner input",
        "",
        "### 1. Zoo ports with no card",
        "",
        "Published and downloadable, undocumented here. Each is either a card to write",
        "or a repo to unpublish. (Bench exports of Apple's own recipes are listed",
        "separately below — those answer to Apple's repo, not to a zoo card.)",
        "",
    ]
    L += [f"- [{r['id']}](https://huggingface.co/{r['id']}) — {r['dl30']} DL/30d, "
          f"{len(r['bundles'])} bundle(s), updated {r['updated']}"
          for r in no_card] or ["- (none)"]

    L += [
        "",
        f"Bench exports of Apple's own recipes, no card expected ({len(official_no_card)}): "
        + ", ".join(f"`{r['id'].split('/')[-1]}`" for r in official_no_card),
        "",
        "### 2. Carded, several bundles, no recipe — which one shipped?",
        "",
        'These are the `status = "unverified"` candidates: the card documents the model,',
        "the repo publishes more than one bundle, and nothing records which is the",
        "published configuration. **Do not guess their `args`.**",
        "",
    ]
    for r in ambiguous:
        L.append(f"- **{r['id']}** ({r['dl30']} DL/30d) — {len(r['bundles'])} bundles:")
        L += [f"  - `{b}`" for b in r["bundles"]]
    if not ambiguous:
        L.append("- (none)")

    unverified = {n: r for n, r in all_recipes().items() if r.get("status") == "unverified"}
    L += [
        "",
        "### 3. Recipes recorded, shipped configuration unknown",
        "",
        f"{len(unverified)} of the {len(all_recipes())} entries in `conversion/recipes.toml` carry",
        '`status = "unverified"`: the script is known, the arguments that produced the published',
        "bundle are not, and nothing in the repo records them. `zoo_convert.py` refuses to run",
        "these without `--force`. Each needs one answer from the owner.",
        "",
    ]
    for name, r in unverified.items():
        # open_questions is stored line-wrapped in the TOML; one question per entry.
        question = " ".join(r.get("open_questions", [])) or "not stated"
        L.append(f"- **`{name}`** ({r.get('hf_repo', '?')}) — {question}")

    L += [
        "",
        "### 4. Carded, exactly one bundle, no recipe",
        "",
        "Unambiguous by construction — the single published bundle *is* the shipped",
        "configuration. These can get a recipe without asking anyone, provided the",
        "export flags are recoverable from the card or the conversion script.",
        "",
    ]
    L += [f"- {r['id']} — `{r['bundles'][0]}`" for r in single] or ["- (none)"]
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="print instead of writing the file")
    ap.add_argument("--offline", action="store_true", help="use only the local cache")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cat = Catalog(**({"cache_dir": args.cache_dir} if args.cache_dir else {}), offline=args.offline)
    rows = collect(cat)
    out = render(rows)
    if args.print:
        print(out)
    else:
        dest = REPO / "models" / "_INVENTORY.md"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(out)
        coreai = [r for r in rows if r["format"] == "coreai"]
        print(f"wrote {dest.relative_to(REPO)} — {len(rows)} repos "
              f"({len(coreai)} Core AI, {sum(len(r['bundles']) for r in coreai)} bundles), "
              f"{sum(1 for r in coreai if r['cards'])} carded, "
              f"{sum(1 for r in rows if r['recipes'])} with a recipe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
