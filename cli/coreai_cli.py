#!/usr/bin/env python3
"""coreai — the zoo's four commands behind one word.

    coreai export  <hf-id | short-name | checkpoint-dir>   which route, and what blocks it
    coreai doctor  <bundle | asset | checkpoint | hf-id>   which known trap it stands on
    coreai verify  <bundle-dir>                            does it compute what the reference computes
    coreai eval    --run <bundle-dir> --task gsm8k         does it still do the job

Each command's own --help has the full story. `pip install coreai-cli` carries the
router, the lint and both gates; converting checkpoints needs Apple's coreai_models
toolchain on the machine, and running a zoo recipe needs the zoo checkout:
https://github.com/john-rocky/coreai-model-zoo

This is a community tool, not an Apple product.
"""
import sys

COMMANDS = ("export", "doctor", "verify", "eval")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("--version", "-V"):
        from importlib.metadata import version
        print("coreai-cli " + version("coreai-cli"))
        return
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        raise SystemExit(0 if argv else 2)
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(f"\nunknown command {cmd!r} — one of: {', '.join(COMMANDS)}")
    module = __import__(f"coreai_{cmd}")
    sys.argv = [f"coreai {cmd}", *rest]
    ret = module.main()
    raise SystemExit(ret if isinstance(ret, int) else None)


if __name__ == "__main__":
    main()
