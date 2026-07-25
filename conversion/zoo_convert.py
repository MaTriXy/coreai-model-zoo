#!/usr/bin/env python3
"""One entrypoint for the zoo's recorded conversion recipes.

    python3 zoo_convert.py list [--unverified]
    python3 zoo_convert.py show <recipe>
    python3 zoo_convert.py doctor                     # check venv + overlay wiring
    python3 zoo_convert.py run <recipe> [--dry-run] [--force] [-- extra script args]

This is a thin index over conversion/export_*.py — it does not convert anything the
scripts can't. It remembers the configuration that produced each published bundle
(models/<family>/recipe.toml), states the prerequisites that configuration depends on,
checks the environment, and runs the right script with the right interpreter.

Two rules it enforces, because both failures are silent:

  - a recipe marked `status = "unverified"` does not run without --force. The repo
    does not record what shipped; running it would produce *a* bundle, not *the*
    bundle, and the difference is invisible afterwards.
  - a missing prerequisite stops the run rather than proceeding. `--dry-run` prints
    everything instead, including the run-time prerequisites (engine patches and
    environment) that the export itself does not need but the resulting bundle does.

Interpreter resolution: --python flag > $ZOO_CONVERT_PYTHON > sibling checkout's
.venv (../../coreai-models/.venv/bin/python) > python3 on PATH. The interpreter must
have the coreai_models package WITH the zoo overlay applied (see overlay/README.md).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODELS = REPO / "models"
# Import that only succeeds when the zoo overlay is applied (not stock upstream).
OVERLAY_CANARY = "coreai_models.models.macos.qwen3_5"


def load_recipes() -> dict:
    """Every models/<family>/recipe.toml, merged. The family is where the card lives."""
    out: dict[str, dict] = {}
    for path in sorted(MODELS.glob("*/recipe.toml")):
        with open(path, "rb") as fh:
            for name, recipe in tomllib.load(fh).items():
                recipe["family"] = path.parent.name
                out[name] = recipe
    return dict(sorted(out.items()))


def resolve_python(flag: str | None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    sibling = REPO.parent / "coreai-models" / ".venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    return shutil.which("python3") or "python3"


def steps_of(recipe: dict) -> list[dict]:
    """Every recipe reduces to a list of steps, single-command ones included."""
    if "steps" in recipe:
        return recipe["steps"]
    step = {k: recipe[k] for k in ("script", "command", "args", "bundle") if k in recipe}
    return [step] if ("script" in step or "command" in step) else []


def build_command(step: dict, python: str, extra: list[str]) -> list[str]:
    if "command" in step:  # stock-exporter recipe: console script from the same venv
        exe = Path(python).parent / step["command"][0]
        return [str(exe) if exe.exists() else step["command"][0], *step["command"][1:], *extra]
    return [python, str(HERE / step["script"]), *step.get("args", []), *extra]


def standalone(step: dict) -> str | None:
    """`uv run` line for a script that declares its own dependencies (PEP 723).

    Those scripts need no venv and no overlay: uv reads the inline block, builds a
    throwaway environment and runs them. Worth printing, because the alternative is a
    four-step checkout-patch-install dance.
    """
    if "script" not in step:
        return None
    path = HERE / step["script"]
    if not path.exists() or "# /// script" not in path.read_text(errors="ignore"):
        return None
    return " ".join(["uv", "run", f"conversion/{step['script']}", *step.get("args", [])])


def missing_prerequisites(recipe: dict) -> list[str]:
    """The prerequisites this tool can check itself: scripts and patch files exist."""
    problems = []
    for step in steps_of(recipe):
        if "script" in step and not (HERE / step["script"]).exists():
            problems.append(f"export script not found: conversion/{step['script']}")
    for patch in recipe.get("runtime_patches", []):
        if not (REPO / patch).exists():
            problems.append(f"runtime patch not found: {patch}")
    return problems


def print_prerequisites(recipe: dict, stream=sys.stdout) -> None:
    def line(label: str, value: str) -> None:
        print(f"{label:<9} {value}", file=stream)

    for step in steps_of(recipe):
        if cmd := standalone(step):
            line("uv", f"{cmd}   (no setup — the script declares its own dependencies)")
    if recipe.get("overlay"):
        line("overlay", "required — coreai_models with conversion/overlay/ applied "
                        "(python3 zoo_convert.py doctor)")
    for need in recipe.get("needs", []):
        line("needs", need)
    for patch in recipe.get("runtime_patches", []):
        line("runtime", f"engine patch {patch}")
    for key, value in (recipe.get("runtime_env") or {}).items():
        line("runtime", f"env {key}={value}")
    if aot := recipe.get("aot"):
        line("device", f"AOT: {aot}")
    for question in recipe.get("open_questions", []):
        line("UNKNOWN", question)


def summarize(recipe: dict) -> str:
    steps = steps_of(recipe)
    if not steps:
        return "(no command recorded)"
    if len(steps) > 1:
        return f"{len(steps)} steps: " + " + ".join(
            Path(s["script"]).name if "script" in s else s["command"][0] for s in steps)
    s = steps[0]
    return (" ".join(s["command"]) if "command" in s
            else f"{s['script']} {' '.join(s.get('args', []))}").strip()


def cmd_list(recipes: dict, show_unverified: bool) -> None:
    rows = [(n, r) for n, r in recipes.items()
            if show_unverified or r.get("status") != "unverified"]
    width = max((len(n) for n, _ in rows), default=0)
    for name, r in rows:
        mark = "?" if r.get("status") == "unverified" else " "
        print(f"{mark} {name:<{width}}  {summarize(r)}")
    if not show_unverified:
        n = sum(1 for r in recipes.values() if r.get("status") == "unverified")
        if n:
            print(f"\n{n} more recipes are unverified — the repo does not record which "
                  f"configuration shipped. See them with `list --unverified`.")


def cmd_show(recipes: dict, name: str, python: str) -> None:
    r = get(recipes, name)
    print(f"recipe    {name}   [{r.get('status', 'unknown')}]")
    family = r.get("family")
    print(f"card      models/{family}/{r.get('card', 'README.md')}" if family else "card      -")
    if repo := r.get("hf_repo"):
        print(f"published https://huggingface.co/{repo}"
              + (f"  ->  {r['bundle']}" if r.get("bundle") else ""))
    for i, step in enumerate(steps_of(r), 1):
        prefix = f"step {i}" if "steps" in r else "command"
        print(f"{prefix:<9} $ " + " ".join(build_command(step, python, [])))
        if step.get("bundle"):
            print(f"{'':11}-> {step['bundle']}")
    print_prerequisites(r)
    for note in r.get("notes", []):
        print(f"note      {note}")


def cmd_doctor(python: str) -> int:
    print(f"python : {python}")
    probe = (
        "import coreai_models, importlib; "
        f"importlib.import_module('{OVERLAY_CANARY}'); "
        "print(coreai_models.__file__)"
    )
    res = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if res.returncode != 0:
        tail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown error"
        print(f"FAIL   : {tail}")
        print("hint   : the interpreter needs coreai_models with the zoo overlay applied —")
        print("         see overlay/README.md (clone pinned base, apply.py, pip install -e python/)")
        return 1
    print(f"package: {res.stdout.strip()}")
    print(f"overlay: OK ({OVERLAY_CANARY} imports)")
    return 0


def get(recipes: dict, name: str) -> dict:
    if name not in recipes:
        print(f"unknown recipe '{name}' — available: {', '.join(recipes)}", file=sys.stderr)
        raise SystemExit(2)
    return recipes[name]


def cmd_run(recipe: dict, name: str, python: str, extra: list[str],
            dry_run: bool, force: bool) -> None:
    steps = steps_of(recipe)
    if not steps:
        print(f"'{name}' records no command — the scripts that produced the published bundles "
              f"are not in this repository:", file=sys.stderr)
        for q in recipe.get("open_questions", []):
            print(f"  - {q}", file=sys.stderr)
        raise SystemExit(2)
    if recipe.get("status") == "unverified" and not force and not dry_run:
        print(f"'{name}' is marked unverified — the repo does not record which configuration "
              f"produced the published bundle:", file=sys.stderr)
        for q in recipe.get("open_questions", []):
            print(f"  - {q}", file=sys.stderr)
        raise SystemExit("refusing to run; --dry-run prints the partial command, --force runs it "
                         "anyway (the result may not match what shipped)")

    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(f"# step {i}/{len(steps)}")
        print("$ " + " ".join(build_command(step, python, extra if i == len(steps) else [])))
    print_prerequisites(recipe, stream=sys.stderr)
    for note in recipe.get("notes", []):
        print(f"# {note}", file=sys.stderr)
    if dry_run:
        return

    if problems := missing_prerequisites(recipe):
        for p in problems:
            print(f"missing prerequisite: {p}", file=sys.stderr)
        raise SystemExit("prerequisites missing — not running the conversion")
    if recipe.get("overlay") and cmd_doctor(python) != 0:
        raise SystemExit("doctor failed — not running the conversion")
    for i, step in enumerate(steps, 1):
        command = build_command(step, python, extra if i == len(steps) else [])
        code = subprocess.run(command, cwd=HERE).returncode
        if code:
            raise SystemExit(f"step {i} failed ({code})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", help="interpreter with coreai_models + overlay installed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--unverified", action="store_true",
                        help="also list recipes whose shipped configuration is not recorded")
    sub.add_parser("doctor")
    p_show = sub.add_parser("show")
    p_show.add_argument("name")
    p_run = sub.add_parser("run")
    p_run.add_argument("name")
    p_run.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    p_run.add_argument("--force", action="store_true", help="run an unverified recipe anyway")
    p_run.add_argument("extra", nargs="*", help="extra args passed through to the script (after --)")
    args = ap.parse_args()

    recipes = load_recipes()
    python = resolve_python(args.python)

    if args.cmd == "list":
        cmd_list(recipes, args.unverified)
    elif args.cmd == "show":
        cmd_show(recipes, args.name, python)
    elif args.cmd == "doctor":
        raise SystemExit(cmd_doctor(python))
    elif args.cmd == "run":
        cmd_run(get(recipes, args.name), args.name, python, args.extra, args.dry_run, args.force)


if __name__ == "__main__":
    main()
