#!/usr/bin/env python3
"""Plan (and optionally run) gate transcripts for models the zoo already publishes.

`coreai_gate.py --transcript` produces the evidence a reader needs to check a published
bundle without rebuilding the fp32 oracle. Models ported before that flag existed have no
transcript, so the cheap re-check path has nothing to check against. This walks the catalog
and fills the gap.

It is **dry-run by default**: it prints exactly what it would do and exits. Gating a model
runs the engine and an fp32 oracle, which loads the machine — pass `--run` only when nothing
else needs it. `--run` also refuses to start while the shared Mac-GPU lock is held, so a
parallel session measuring energy or throughput is not disturbed.

    python3 conversion/backfill_gate_transcripts.py              # the plan
    python3 conversion/backfill_gate_transcripts.py --run        # do it
    python3 conversion/backfill_gate_transcripts.py --run --only qwen3.5-2b

A transcript records what a run actually produced. Nothing here reconstructs one from a
card's prose after the fact — a model that cannot be re-gated stays without a transcript,
and its card keeps saying what was gated instead of pretending to prove it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import exports_dir, gpu_lock, repo_root  # noqa: E402
from _recipe import revision, source_model  # noqa: E402

import coreai_gate  # noqa: E402


# CoreAIKit's on-disk cache: <root>/<hf-owner>/<hf-repo>/<revision>/<bundle path>. Gating here
# is strictly better than gating a local export — it is the artifact apps actually download, at
# the revision the catalog pins, so the transcript describes what shipped rather than what was
# built. Overridable because the default is a macOS app-support path.
MODELSTORE = Path(
    os.environ.get("ZOO_MODELSTORE",
                   Path.home() / "Library" / "Application Support" / "CoreAIKit" / "Models"))


def locate_bundle(hf_repo: str | None, bundle: str) -> tuple[Path | None, str]:
    """Published copy first, local export second. Returns (path, provenance)."""
    if hf_repo:
        for rev_dir in sorted((MODELSTORE / hf_repo).glob("*")) if (MODELSTORE / hf_repo).is_dir() else []:
            candidate = rev_dir / bundle
            if candidate.is_dir():
                return candidate, f"published {hf_repo}@{rev_dir.name[:12]}"
    local = exports_dir() / bundle
    return (local, "local export") if local.is_dir() else (None, "")


def plan_model(model_dir: Path) -> list[dict]:
    """One entry per published bundle in this model's recipe."""
    recipe_path = model_dir / "recipe.toml"
    if not recipe_path.exists():
        return []
    recipe = tomllib.loads(recipe_path.read_text())
    out: list[dict] = []
    # A recipe file is keyed by bundle name; a model may publish more than one.
    for name, step in recipe.items():
        if not isinstance(step, dict) or "bundle" not in step:
            continue
        hf_id, hf_id_from = source_model(step)
        if not hf_id:
            out.append({"model": name, "skip": "no source model in the recipe or its exporter"})
            continue
        arch = coreai_gate.detect_arch(step["bundle"], hf_id)
        if not arch:
            out.append({"model": name, "skip": "no oracle for this architecture"})
            continue
        bundle, provenance = locate_bundle(step.get("hf_repo"), step["bundle"])
        if bundle is None:
            out.append({"model": name, "skip": "bundle not on this machine"})
            continue
        out.append({
            "model": name,
            "dir": model_dir.name,
            "arch": arch,
            "hf_id": hf_id,
            "hf_id_from": hf_id_from,
            "revision": revision(step),
            "bundle": str(bundle),
            "provenance": provenance,
            "status": step.get("status"),
            "out": str(model_dir / f"gate-{name}.json"),
        })
    if not out:
        return [{"model": model_dir.name, "skip": "recipe records no bundle"}]
    return out


def command_for(job: dict) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).with_name("coreai_gate.py")),
           job["bundle"], job["hf_id"], "--arch", job["arch"],
           "--transcript", job["out"]]
    if job.get("revision"):
        cmd += ["--revision", job["revision"]]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true",
                    help="actually gate (loads the machine); default is to print the plan")
    ap.add_argument("--only", metavar="MODEL", help="restrict to one models/<MODEL> directory")
    ap.add_argument("--force", action="store_true", help="re-gate models that already have one")
    args = ap.parse_args()

    models = sorted(p for p in (repo_root() / "models").iterdir() if p.is_dir())
    if args.only:
        models = [p for p in models if p.name == args.only] or sys.exit(f"no models/{args.only}")

    jobs, skipped = [], []
    for d in models:
        for job in plan_model(d):
            if "skip" in job:
                skipped.append(job)
            elif Path(job["out"]).exists() and not args.force:
                skipped.append({"model": job["model"], "skip": "already has a transcript"})
            else:
                jobs.append(job)

    print(f"{len(jobs)} bundle(s) can be gated, {len(skipped)} skipped\n")
    for j in jobs:
        print(f"  {j['model']:<24} arch={j['arch']:<12} {j['hf_id']}")
        print(f"  {'':24} src: {j['provenance']}")
        print(f"  {'':24} -> {Path(j['out']).relative_to(repo_root())}")
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print(f"  {s['model']:<24} {s['skip']}")

    if not args.run:
        print("\nThis was a plan. Re-run with --run when the machine is free.")
        return

    lock = gpu_lock()
    if lock.exists():
        sys.exit(f"\nthe shared GPU lock is held ({lock}) — another session is using the machine.\n"
                 "  Gating now would perturb its measurements. Wait, or clear the lock if stale.")

    failures = []
    for j in jobs:
        print(f"\n=== {j['model']}")
        if subprocess.run(command_for(j)).returncode != 0:
            failures.append(j["model"])
    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} transcripts written")
    if failures:
        sys.exit("failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
