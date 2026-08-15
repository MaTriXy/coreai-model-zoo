#!/usr/bin/env python3
"""coreai export — find the export route for a model, or say plainly that there isn't one.

    coreai_export.py <hf-id | short-name | checkpoint-dir> [--device mac|iphone] [--run]
    coreai_export.py --list                  # the whole support matrix
    coreai_export.py --list --device iphone  # only what has an iOS path

This is a **router, not an exporter**. It does not convert anything itself. What it does is
answer, before you spend an hour finding out the hard way:

  1. does this model have a route at all, and through which backend;
  2. if it is an iPhone target, does that route have an iOS path (most do not);
  3. what exactly is unvalidated about the route it picked;
  4. what the resolved command is.

There are three backends and they are not interchangeable:

  preset   Apple's stock exporter WITH a named preset for this exact checkpoint.
           Compute precision, compression and context length are all resolved for you,
           and the combination has been run by Apple.
  generic  Apple's stock exporter routing by HF `model_type` only. It will run. Nothing
           about the recipe has been validated for THIS checkpoint — quantization
           tolerance is a per-model property, so this is a starting point, not an answer.
  zoo      A recorded community recipe for this family. Reproduces a bundle that shipped.

  none     The `model_type` does not route. This is not a CLI problem: a new architecture
           needs a re-authored model class. Saying so is the useful output.

By default it prints the plan and stops. `--run` executes. That default is deliberate:
this tool cannot tell you the bundle it would produce is correct, and the gate that can
is a separate step it prints for you.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Device profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    name: str
    platform: str  # the exporter's --platform value
    aot_arch: str | None
    note: str


# Architecture names track the DEVICE IDENTIFIER major version, not the marketing name
# (iPhone 17 Pro = iPhone18,1 -> h18p; an M-series Mac = Mac16,x -> h16c), and
# `coreai-build compile` exits 0 for any arch you name — only a device load validates it.
# So only the two archs this project has actually loaded on hardware are offered.
DEVICES: dict[str, Device] = {
    "mac": Device("mac", "macOS", "h16c",
                  "h16c is the only macOS arch that loads on an M4 Max here; the other 19 raise."),
    "iphone": Device("iphone", "iOS", "h18p",
                     "h18p = iPhone18,1 = iPhone 17 Pro. A different iPhone needs its own arch, "
                     "and an h17p bundle pushed to an iPhone18,1 fails with invalidCompiledModel."),
}
DEVICE_ALIASES = {"iphone-16gb": "iphone", "iphone-8gb": "iphone", "ios": "iphone", "macos": "mac"}

# knowledge/aot-and-specialization.md and knowledge/pipelined-engine.md thresholds.
JIT_WATCH_BYTES = 1 << 30
ENTITLEMENT_BYTES = 2 * (1 << 30)


# ---------------------------------------------------------------------------
# Apple's stock tables, read from the installed package
# ---------------------------------------------------------------------------

# Snapshot of coreai_models 2026-07-31, used only when the package cannot be imported.
# `--verify-tables` re-derives these and diffs, so a stale snapshot is loud, not silent.
SNAPSHOT = {
    "registry": {
        "gemma3_text": [True, False], "gemma4_text": [True, False], "gpt_oss": [True, False],
        "mistral": [True, True], "mixtral": [True, False], "mistral3": [True, False],
        "olmo2": [True, True], "phi3": [True, True], "smollm3": [True, True],
        "qwen2": [True, True], "qwen3": [True, True], "qwen3_moe": [True, False],
        "qwen3_5_text": [True, False], "qwen3_vl": [True, False],
    },
    "remapping": {
        "gemma3": "gemma3_text", "qwen2_5": "qwen2", "llama": "mistral",
        "qwen3_5": "qwen3_5_text", "gemma4": "gemma4_text",
    },
}

PROBE = r"""
import json
from coreai_models.models.registry import _get_registry, MODEL_TYPE_REMAPPING
reg = {k: [v.macos_class is not None, v.ios_class is not None]
       for k, v in _get_registry().items()}
print("<<<JSON>>>" + json.dumps({"registry": reg, "remapping": dict(MODEL_TYPE_REMAPPING)}))
"""


def find_zoo_root() -> Path | None:
    """The zoo checkout, from wherever this file lives.

    A zoo checkout is the directory holding both `models/` and `conversion/`. Walking up
    finds it when the CLI lives inside the zoo; the explicit siblings cover the CLI living
    beside it. Both layouts are supported on purpose — placement is not settled.
    """
    for base in (HERE, *HERE.parents):
        if (base / "models").is_dir() and (base / "conversion").is_dir():
            return base
    for candidate in (Path.home() / "code/coreai/coreai-models-community",
                      HERE.parent / "coreai-model-zoo"):
        if (candidate / "models").is_dir():
            return candidate
    return None


def find_coreai_models() -> Path | None:
    """Apple's coreai-models checkout — a sibling of the zoo, or of the CLI."""
    roots = [find_zoo_root(), HERE, Path.home() / "code/coreai"]
    for base in filter(None, roots):
        for candidate in (base.parent / "coreai-models", base / "coreai-models"):
            if (candidate / "python").is_dir() or (candidate / ".venv").is_dir():
                return candidate
    return None


def resolve_python(flag: str | None) -> str:
    """Same resolution order as conversion/zoo_convert.py, so both agree on the venv."""
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    if (base := find_coreai_models()) and (venv := base / ".venv/bin/python").exists():
        return str(venv)
    return shutil.which("python3") or "python3"


def apple_tables(python: str) -> tuple[dict, str]:
    """(tables, provenance). Falls back to the dated snapshot, and says so."""
    res = subprocess.run([python, "-c", PROBE], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("<<<JSON>>>"):
            return json.loads(line[len("<<<JSON>>>"):]), f"live ({python})"
    return SNAPSHOT, "SNAPSHOT 2026-07-31 (coreai_models not importable — may be stale)"


def apple_presets(python: str) -> list[dict]:
    res = subprocess.run(
        [python, "-m", "coreai_models.model_registry", "--list-models", "--type", "llm", "--json"],
        capture_output=True, text=True,
    )
    try:
        rows = json.loads(res.stdout)
    except ValueError:
        return []
    return rows if isinstance(rows, list) else rows.get("models", [])


# ---------------------------------------------------------------------------
# The zoo's recorded recipes
# ---------------------------------------------------------------------------



def zoo_recipes() -> tuple[dict[str, dict], Path | None, str]:
    """Every models/<family>/recipe.toml, keyed by recipe name, plus the source hf id.

    Without a checkout — a pip install — the bundled snapshot still answers the routing
    question. *Running* a zoo recipe needs the checkout either way, and main() says so
    rather than executing a command that has no working directory to run in.
    """
    root = find_zoo_root()
    if root is None:
        try:
            import coreai_zoo_routes
        except ImportError:
            return {}, None, "no zoo checkout and no bundled snapshot"
        return (coreai_zoo_routes.ROUTES, None,
                f"SNAPSHOT {coreai_zoo_routes.DATE} (no zoo checkout — may be stale)")
    out: dict[str, dict] = {}
    for path in sorted((root / "models").glob("*/recipe.toml")):
        with open(path, "rb") as fh:
            for name, recipe in tomllib.load(fh).items():
                recipe["family"] = path.parent.name
                kind, ident = recipe_source(recipe, root)
                recipe["source_kind"], recipe["source_hf_id"] = kind, (ident if kind == "hf" else None)
                recipe["source_id"] = ident
                out[name] = recipe
    return out, root, f"live ({root})"


HF_URL = re.compile(r"huggingface\.co/([A-Za-z0-9][\w.\-]*/[\w.\-]+)")
GH_URL = re.compile(r"github\.com/([A-Za-z0-9][\w.\-]*/[\w.\-]+)")
HF_CONST = re.compile(
    r'(?:default\s*=\s*|HF_ID\s*=\s*|MODEL_ID\s*=\s*|REPO\s*=\s*)["\']([\w.\-]+/[\w.\-]+)["\']')
# The zoo's own output repos and the toolchain repos are never the SOURCE.
NOT_A_SOURCE = ("mlboydaisuke/", "john-rocky/", "apple/coreai", "huggingface/")


def recipe_source(recipe: dict, root: Path) -> tuple[str, str | None]:
    """(kind, id) for the model a recipe converts FROM.

    `hf_repo` in recipe.toml is the OUTPUT repo, so it cannot stand in.

    `source_hf_id` / `source_repo` are the recipe's own declaration and win outright. The
    recovery below is the fallback for a recipe written before those fields existed, or
    added without them — it reads the three other places the source is recorded rather
    than reporting "unknown" for a recipe that merely puts it somewhere else:

      1. the recipe's own args (--hf-id)
      2. the export script's argparse default or module constant
      3. the model card's first non-zoo huggingface.co link
      4. the model card's first non-toolchain github.com link — for ports whose upstream
         is not on the Hub at all (RF-DETR, YOLOX, AdcSR). Those are not a gap in the
         index; they are simply not addressable by an HF id.
    """
    if declared := recipe.get("source_hf_id"):
        return "hf", declared
    if declared := recipe.get("source_repo"):
        return "github", declared.removeprefix("github.com/")

    steps = recipe.get("steps") or [recipe]
    for step in steps:
        args = [str(a) for a in step.get("args", [])]
        for key in ("--hf-id", "--model", "--source-hf-id"):
            if key in args:
                return "hf", args[args.index(key) + 1]

    for step in steps:
        script = step.get("script")
        if script and (root / "conversion" / script).exists():
            text = (root / "conversion" / script).read_text(errors="ignore")
            for m in HF_CONST.finditer(text):
                if not m.group(1).startswith(NOT_A_SOURCE):
                    return "hf", m.group(1)

    card = root / "models" / recipe.get("family", "") / recipe.get("card", "README.md")
    if card.exists():
        text = card.read_text(errors="ignore")
        for m in HF_URL.finditer(text):
            if not m.group(1).startswith(NOT_A_SOURCE):
                return "hf", m.group(1)
        for m in GH_URL.finditer(text):
            if not m.group(1).startswith(NOT_A_SOURCE):
                return "github", m.group(1)
    return "unknown", None


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def read_model_type(target: str, python: str) -> tuple[str | None, Path | None, str]:
    """(model_type, local config dir, how it was resolved)."""
    path = Path(target)
    if path.is_dir() and (path / "config.json").exists():
        cfg = json.loads((path / "config.json").read_text())
        return cfg.get("model_type"), path, "local config.json"
    if HF_ID.match(target):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        code = (
            "import json,sys\n"
            "from huggingface_hub import hf_hub_download\n"
            f"p = hf_hub_download({target!r}, 'config.json')\n"
            "print('<<<CFG>>>' + p)\n"
        )
        res = subprocess.run([python, "-c", code], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("<<<CFG>>>"):
                p = Path(line[len("<<<CFG>>>"):])
                cfg = json.loads(p.read_text())
                return cfg.get("model_type"), p.parent, f"fetched config.json for {target}"
        return None, None, f"could not fetch config.json ({res.stderr.strip().splitlines()[-1:]})"
    return None, None, "not an HF id and not a checkpoint directory"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass
class Route:
    backend: str  # preset | generic | zoo | none
    command: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)


def route(target: str, model_type: str | None, device: Device, tables: dict,
          presets: list[dict], recipes: dict, python: str, out_dir: str | None,
          zoo_root: Path | None) -> Route:
    registry, remap = tables["registry"], tables["remapping"]
    resolved = remap.get(model_type or "", model_type)

    # --- a recorded zoo recipe beats a generic route: it reproduces a bundle that shipped
    zoo_hits = [n for n, r in recipes.items() if r.get("source_hf_id") == target]
    family_hits = [n for n, r in recipes.items()
                   if not zoo_hits and target.split("/")[-1].lower().startswith(r["family"].split("-")[0])]

    # --- Apple preset for this exact checkpoint + platform?
    hit = next((p for p in presets
                if (p.get("hf_id") == target or p.get("short_name") == target)
                and p.get("variant") == device.platform), None)
    other_platform = [p.get("variant") for p in presets
                      if p.get("hf_id") == target or p.get("short_name") == target]

    if hit:
        r = Route("preset", [
            python, "-m", "coreai_models.llm.export", hit["short_name"],
            "--platform", device.platform,
        ])
        r.caveats.append(
            f"named preset '{hit['short_name']}' — compute precision, compression and context "
            f"length all come from the preset, and Apple has run this combination."
        )
    elif resolved in registry:
        macos_ok, ios_ok = registry[resolved]
        r = Route("generic", [
            python, "-m", "coreai_models.llm.export", target,
            "--platform", device.platform,
            "--compute-precision", "float16",
            "--experimental",
        ])
        r.caveats.append(
            f"model_type {model_type!r}"
            + (f" (remapped to {resolved!r})" if resolved != model_type else "")
            + " routes, but there is NO preset for this checkpoint. --experimental is required, "
              "and nothing about the recipe has been validated for these weights."
        )
        r.caveats.append(
            "Quantization tolerance is a per-model property, not an architecture property: the "
            "same int4-linear recipe is top-1 exact on Gemma-4 and Ornith-9B and fails the gate "
            "on Qwen3.5-0.8B/2B and LFM2.5. Expect to gate, not to ship."
        )
        if device.platform == "iOS" and not ios_ok:
            r.blockers.append(
                f"{resolved!r} has a macOS re-authoring but NO iOS one, so the stock exporter "
                f"has no iOS path for it. This is the single most common wasted hour: the "
                f"macOS export succeeds and there is simply nothing to target the phone with."
            )
        if device.platform == "macOS" and not macos_ok:
            r.blockers.append(f"{resolved!r} has no macOS re-authoring.")
    else:
        r = Route("none")
        r.blockers.append(
            f"model_type {model_type!r} is not in Apple's registry and has no remapping. "
            f"There is no route. A new architecture needs a re-authored model class in "
            f"coreai_models/models/macos/ — that is a port, not a flag, and it is exactly why "
            f"the zoo carries 57 export scripts instead of one."
        )

    if other_platform and not hit:
        r.caveats.append(
            f"a preset for this model exists, but only for: {', '.join(sorted(set(other_platform)))}"
        )
    if zoo_hits:
        r.caveats.append(
            f"the zoo has a recorded recipe for this exact source model: {', '.join(zoo_hits)}. "
            f"That reproduces a bundle that shipped and gated — prefer it over a generic route: "
            f"python3 conversion/zoo_convert.py show {zoo_hits[0]}"
        )
        if r.backend in ("generic", "none"):
            r.backend = "zoo"
            r.command = ["python3", "conversion/zoo_convert.py", "run", zoo_hits[0]]
            # The zoo route is a real route, so Apple's limitations stop being blockers and
            # become context. Leaving them as blockers would print "BLOCKED" above a command.
            r.caveats = [f"Apple's stock exporter has no route here — {b}" for b in r.blockers] \
                + r.caveats
            r.blockers = []
            if zoo_root is None:
                r.caveats.append(
                    "this route comes from the bundled snapshot — running it needs the zoo "
                    "checkout: git clone https://github.com/john-rocky/coreai-model-zoo, "
                    "then run the command from that directory."
                )
            recipe = recipes[zoo_hits[0]]
            if recipe.get("status") == "unverified":
                r.blockers.append(
                    f"recipe {zoo_hits[0]!r} is marked unverified: the repo does not record which "
                    f"configuration produced the published bundle, so running it yields *a* "
                    f"bundle, not *the* bundle — and the difference is invisible afterwards. "
                    f"zoo_convert refuses without --force, and so does this."
                )
            for need in recipe.get("needs", []):
                r.caveats.append(f"recipe prerequisite: {need}")
            if recipe.get("overlay"):
                r.caveats.append(
                    "this recipe needs coreai_models with the zoo overlay applied — check with "
                    "python3 conversion/zoo_convert.py doctor")
    elif family_hits and r.backend == "none":
        r.caveats.append(
            "no recipe records this source id, but the zoo has recipes whose family name is "
            "close: " + ", ".join(sorted(family_hits)[:4]) + " (a name hint, not a match — only "
            "13 of 65 recipes record their source model at all)"
        )

    if out_dir and r.backend in ("preset", "generic"):
        r.command += ["--output-dir", out_dir]

    if device.aot_arch and r.backend in ("preset", "generic", "zoo"):
        r.followups.append(
            f"AOT, if the cold specialization stalls or aborts: xcrun coreai-build compile "
            f"<bundle>.aimodel --output <dir> --platform {device.platform} "
            f"--architecture {device.aot_arch} --preferred-compute gpu "
            f"--min-deployment-version 27.0"
        )
    if r.backend == "none":
        return r
    r.followups.append(
        "Gate before you believe it: python3 conversion/coreai_gate.py <bundle> <hf-id> -n 16 "
        "— token-exact against the HF oracle on a prompt whose fp32 top-2 margin clears 0.1 at "
        "every position. An export that has not been gated is not a port."
    )
    doctor = ("coreai doctor" if Path(sys.argv[0]).name.startswith("coreai ")
              else f"python3 {HERE / 'coreai_doctor.py'}")
    r.followups.append(f"Lint the result: {doctor} <bundle> "
                       f"--profile {'iphone' if device.platform == 'iOS' else 'mac'}")
    return r


# ---------------------------------------------------------------------------
# Doctor pre-flight
# ---------------------------------------------------------------------------


def preflight(config_dir: Path | None, target: str) -> list[tuple[str, str, str]]:
    """Run doctor's checkpoint rules. Returns (severity, id, evidence)."""
    if config_dir is None:
        return []
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.path.insert(0, str(HERE))
    try:
        import coreai_doctor as doctor
    except ImportError:
        return []
    rep = doctor.Report(target=target, kind="checkpoint")
    try:
        doctor.check_checkpoint(config_dir, rep, 32, target if HF_ID.match(target) else None)
    except Exception as exc:  # a lint must never be the thing that stops an export
        return [("info", "DOCTOR-PREFLIGHT-FAILED", f"{type(exc).__name__}: {exc}")]
    return [(f.rule.severity, f.rule.id, f.evidence) for f in rep.findings]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

BACKEND_LINE = {
    "preset": "Apple stock exporter, named preset for this checkpoint",
    "generic": "Apple stock exporter, routed by model_type only — UNVALIDATED for this checkpoint",
    "zoo": "zoo recorded recipe — reproduces a bundle that shipped",
    "none": "no route",
}


def print_matrix(tables: dict, presets: list[dict], device: Device | None) -> None:
    registry, remap = tables["registry"], tables["remapping"]
    by_type: dict[str, list[str]] = {}
    for p in presets:
        key = p.get("hf_id", "?")
        by_type.setdefault(key, []).append(p.get("variant") or "?")

    print("Apple's stock exporter — every model_type it accepts\n")
    print(f"  {'model_type':<16}{'macOS':<8}{'iOS':<8}reached also by")
    aliases: dict[str, list[str]] = {}
    for src, dst in remap.items():
        aliases.setdefault(dst, []).append(src)
    for mt in sorted(registry):
        macos_ok, ios_ok = registry[mt]
        if device and device.platform == "iOS" and not ios_ok:
            continue
        print(f"  {mt:<16}{'yes' if macos_ok else '—':<8}{'yes' if ios_ok else '—':<8}"
              f"{', '.join(sorted(aliases.get(mt, []))) or '—'}")
    n_ios = sum(1 for v in registry.values() if v[1])
    print(f"\n  {len(registry)} model types, {n_ios} with an iOS path. "
          f"{len(remap)} more HF model_type values remap in (note `llama` -> `mistral`, "
          f"which is why most plain Llama checkpoints route).")

    print(f"\nNamed presets ({len(presets)}) — the only combinations Apple has actually run\n")
    for hf_id in sorted(by_type):
        variants = sorted(set(by_type[hf_id]))
        if device and device.platform not in variants:
            continue
        print(f"  {hf_id:<52}{', '.join(variants)}")
    print("\nEverything else is either a generic (unvalidated) route or no route at all.")


def render(target: str, model_type: str | None, how: str, device: Device, r: Route,
           findings: list[tuple[str, str, str]], provenance: str) -> None:
    print(f"coreai export  {target}")
    print(f"model_type     {model_type!r}   ({how})")
    print(f"device         {device.name} -> --platform {device.platform}, AOT arch {device.aot_arch}")
    print(f"tables         {provenance}")
    print(f"route          {r.backend.upper()} — {BACKEND_LINE[r.backend]}")
    print()

    if findings:
        print("--- CHECKPOINT PRE-FLIGHT (coreai doctor) " + "-" * 30)
        for sev, rid, ev in sorted(findings, key=lambda f: f[0]):
            print(f"  {sev.upper():<9}{rid}")
            print(f"            {ev}")
        print()

    if r.blockers:
        print("--- BLOCKED " + "-" * 60)
        for b in r.blockers:
            print(f"  {b}")
        print()

    if r.caveats:
        print("--- WHAT IS AND IS NOT SETTLED " + "-" * 42)
        for c in r.caveats:
            print(f"  - {c}")
        print()

    if r.command:
        print("--- COMMAND " + "-" * 60)
        print("  $ " + " ".join(r.command))
        print()

    if r.backend != "none":
        print("--- AFTER THE EXPORT " + "-" * 51)
        for f in r.followups:
            print(f"  - {f}")
        print()
        print(f"  {device.note}")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="HF id, Apple preset short-name, or checkpoint dir")
    ap.add_argument("--device", default="mac", help="mac | iphone (aliases: macos, ios, iphone-8gb, iphone-16gb)")
    ap.add_argument("--out", default=None, help="--output-dir for the exporter")
    ap.add_argument("--python", default=None, help="interpreter with coreai_models installed")
    ap.add_argument("--run", action="store_true", help="execute the command instead of printing it")
    ap.add_argument("--list", action="store_true", help="print the support matrix and exit")
    ap.add_argument("--verify-tables", action="store_true",
                    help="diff the live coreai_models tables against this file's snapshot")
    ap.add_argument("--json", action="store_true", help="machine-readable route")
    args = ap.parse_args()

    python = resolve_python(args.python)
    tables, provenance = apple_tables(python)

    if args.verify_tables:
        drift = {k: (SNAPSHOT[k], tables[k]) for k in SNAPSHOT if SNAPSHOT[k] != tables.get(k)}
        print(f"tables: {provenance}")
        if not drift:
            print("snapshot matches the installed coreai_models — no drift")
            return
        for key, (old, new) in drift.items():
            print(f"\n{key} DRIFTED")
            print(f"  snapshot: {json.dumps(old, sort_keys=True)}")
            print(f"  installed: {json.dumps(new, sort_keys=True)}")
        raise SystemExit(1)

    device_key = DEVICE_ALIASES.get(args.device, args.device)
    if device_key not in DEVICES:
        ap.error(f"unknown device {args.device!r}; known: {', '.join(DEVICES)} "
                 f"(+ aliases {', '.join(DEVICE_ALIASES)})")
    device = DEVICES[device_key]
    presets = apple_presets(python)

    if args.list:
        print_matrix(tables, presets, device if args.device != "mac" else None)
        return
    if not args.target:
        ap.error("a target is required (or --list)")

    model_type, config_dir, how = read_model_type(args.target, python)
    # An Apple short-name is not an HF id and has no local config; resolve it through the preset.
    if model_type is None:
        named = next((p for p in presets if p.get("short_name") == args.target), None)
        if named:
            model_type, config_dir, how = read_model_type(named["hf_id"], python)
            how += f" (via preset short-name {args.target!r})"

    recipes, zoo_root, zoo_prov = zoo_recipes()
    if zoo_root is None:
        provenance += f"; zoo routes {zoo_prov}"
    r = route(args.target, model_type, device, tables, presets, recipes, python, args.out,
              zoo_root)
    findings = preflight(config_dir, args.target)

    if args.json:
        print(json.dumps({
            "target": args.target, "model_type": model_type, "device": device.name,
            "platform": device.platform, "aot_arch": device.aot_arch, "backend": r.backend,
            "command": r.command, "caveats": r.caveats, "blockers": r.blockers,
            "followups": r.followups, "tables": provenance,
            "preflight": [{"severity": s, "id": i, "evidence": e} for s, i, e in findings],
        }, indent=2))
    else:
        render(args.target, model_type, how, device, r, findings, provenance)

    fatal = [f for f in findings if f[0] in ("fatal", "silent")]
    if args.run:
        if r.blockers:
            raise SystemExit("refusing to run: see BLOCKED above")
        if r.backend == "zoo" and zoo_root is None:
            raise SystemExit(
                "refusing to run: this is a zoo recipe and there is no zoo checkout to run "
                "it in. git clone https://github.com/john-rocky/coreai-model-zoo, then from "
                "that directory: " + " ".join(r.command))
        if fatal:
            raise SystemExit(
                "refusing to run: the checkpoint pre-flight found a fatal or silent-corruption "
                "pattern. Read it, then re-run with the finding addressed — or without --run to "
                "get the command and decide for yourself.")
        cwd = zoo_root if r.backend == "zoo" else None
        # The plan must land before the subprocess writes to the same fd, or the export's
        # own logging arrives first and the reader sees the command after its output.
        sys.stdout.flush()
        raise SystemExit(subprocess.run(r.command, cwd=cwd).returncode)

    raise SystemExit(2 if r.blockers else (1 if fatal else 0))


if __name__ == "__main__":
    main()
