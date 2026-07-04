#!/usr/bin/env python3
"""One entrypoint for the zoo's verified conversion recipes.

    python3 zoo_convert.py list
    python3 zoo_convert.py show <recipe>
    python3 zoo_convert.py doctor            # check venv + overlay wiring
    python3 zoo_convert.py run <recipe> [--dry-run] [-- extra script args]

This is a thin index over conversion/export_*.py — it does not convert anything the
scripts can't, it just remembers the verified flags (recipes.toml), checks the
environment first, and runs the right script with the right interpreter.

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
RECIPES = HERE / "recipes.toml"
# Import that only succeeds when the zoo overlay is applied (not stock upstream).
OVERLAY_CANARY = "coreai_models.models.macos.qwen3_5"


def load_recipes() -> dict:
    with open(RECIPES, "rb") as fh:
        return tomllib.load(fh)


def resolve_python(flag: str | None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    sibling = HERE.parent.parent / "coreai-models" / ".venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    return shutil.which("python3") or "python3"


def build_command(name: str, recipe: dict, python: str, extra: list[str]) -> list[str]:
    if "command" in recipe:  # stock-exporter recipe: console script from the same venv
        exe = Path(python).parent / recipe["command"][0]
        return [str(exe) if exe.exists() else recipe["command"][0], *recipe["command"][1:], *extra]
    return [python, str(HERE / recipe["script"]), *recipe.get("args", []), *extra]


def cmd_list(recipes: dict) -> None:
    width = max(len(n) for n in recipes)
    for name, r in recipes.items():
        summary = " ".join(r["command"]) if "command" in r else f"{r['script']} {' '.join(r.get('args', []))}"
        print(f"{name:<{width}}  {summary}")


def cmd_show(recipes: dict, name: str) -> None:
    r = get(recipes, name)
    print(f"recipe : {name}")
    if "command" in r:
        print(f"command: {' '.join(r['command'])}")
    else:
        print(f"script : {r['script']}")
        print(f"args   : {' '.join(r.get('args', []))}")
    print(f"card   : {r.get('card', '-')}")
    for note in r.get("notes", []):
        print(f"note   : {note}")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", help="interpreter with coreai_models + overlay installed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("doctor")
    p_show = sub.add_parser("show")
    p_show.add_argument("name")
    p_run = sub.add_parser("run")
    p_run.add_argument("name")
    p_run.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    p_run.add_argument("extra", nargs="*", help="extra args passed through to the script (after --)")
    args = ap.parse_args()

    recipes = load_recipes()
    python = resolve_python(args.python)

    if args.cmd == "list":
        cmd_list(recipes)
    elif args.cmd == "show":
        cmd_show(recipes, args.name)
    elif args.cmd == "doctor":
        raise SystemExit(cmd_doctor(python))
    elif args.cmd == "run":
        recipe = get(recipes, args.name)
        command = build_command(args.name, recipe, python, args.extra)
        print("$ " + " ".join(command))
        for note in recipe.get("notes", []):
            print(f"# {note}", file=sys.stderr)
        if args.dry_run:
            return
        if cmd_doctor(python) != 0:
            raise SystemExit("doctor failed — not running the conversion")
        raise SystemExit(subprocess.run(command, cwd=HERE).returncode)


if __name__ == "__main__":
    main()
