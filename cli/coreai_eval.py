#!/usr/bin/env python3
"""coreai eval — task accuracy for a port, and the protocol discipline to compare two.

    coreai_eval.py --score gen.json --task gsm8k --arm "iphone int8"   # score a run
    coreai_eval.py --compare a.json b.json                             # compare two runs
    coreai_eval.py --tasks                                             # what ships built in

WHY THIS IS NOT PART OF `verify`
--------------------------------
`verify` asks whether the bundle computes what the reference computes. That is the right
question and it has a blind spot the notes state plainly: **an equivalence gate cannot
detect a defect its reference shares.** The case that produced this file: identical weights,
LiteRT with int8 activations scored 85/100 on GSM8K and the same weights with fp16
activations scored 48/100. Token-exact against an fp16 oracle passes all day.

So this is the other question — does the port still do the job — and it is the one a client
asks. `verify` gates the export. `eval` gates the *product*.

WHY MOST OF THIS FILE IS ABOUT COMPARING RATHER THAN SCORING
------------------------------------------------------------
Scoring is thirty lines. The expensive part is that a task number means nothing on its own
and almost nothing next to a number produced under a different protocol. This project has
now published a wrong conclusion from that twice:

  * A "12-point quality gap" between two runtimes that was a 600-token generation budget on
    one side and 2048 on the other. Same weights, same mode, different amount of room to
    finish the answer. The models were fine; the comparison was not.
  * A quantization blamed for a quality loss before the arms had been matched at all.

Both are invisible in the number and obvious in the configuration. So an arm here records
its configuration, `--compare` **refuses to print a delta** until the arms agree on the
fields that decide the answer, and the truncation rate is reported next to every score
whether you asked for it or not — because equal budgets do not mean equal truncation, and
an arm that gets cut off more is being measured on a different task.

BRING YOUR OWN TASK
-------------------
`--task path/to/task.json` takes the same shape as the built-ins. A client's own eval set
is the point: the harness is generic, the questions are theirs.

Stdlib only. Scoring needs no model, no GPU and no device, so a device run and a Mac run are
scored by exactly the same code — which is the only way their numbers are comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

# A task is data plus four decisions: how to ask, how to read the answer out of free text,
# how to normalise both sides, and what counts as a match. Keeping them declarative is what
# lets someone drop in their own set without touching this file.
BUILTIN_TASKS: dict[str, dict] = {
    "gsm8k": {
        "name": "gsm8k",
        "description": "Grade-school math word problems, exact-match on the final number.",
        # Searched in order; override with --data. Deliberately not vendored: a dataset does
        # not belong in a code repository.
        "data_candidates": [
            "~/code/litertlm-convert/evaldata/gsm8k_test.jsonl",
            "~/code/apple-silicon-llm-bench/evaldata/gsm8k_test.jsonl",
            "./evaldata/gsm8k_test.jsonl",
        ],
        "input_field": "question",
        "gold_field": "answer",
        # GSM8K gold is a worked solution ending in "#### <number>".
        "gold_extract": r"####\s*(.+)\s*$",
        "instruction": (
            "\n\nSolve this step by step. After your reasoning, write the final answer on "
            "its own line in the exact form:\n#### <number>"
        ),
        "pred_extract": r"####\s*([\-0-9\.,]+)",
        # The marker the model was asked to emit. Its absence is the truncation signal.
        "answer_marker": "####",
        "normalize": "number",
        "match": "exact",
    }
}


def load_task(spec: str) -> dict:
    """A built-in name or a path to a JSON task file."""
    if spec in BUILTIN_TASKS:
        return dict(BUILTIN_TASKS[spec])
    path = Path(spec).expanduser()
    if not path.exists():
        raise SystemExit(
            f"unknown task {spec!r}: not a built-in ({', '.join(BUILTIN_TASKS)}) "
            f"and no file at {path}")
    task = json.loads(path.read_text())
    task.setdefault("name", path.stem)
    return task


def resolve_data(task: dict, override: str | None) -> Path:
    candidates = [override] if override else task.get("data_candidates", [])
    if task.get("data"):
        candidates = [task["data"]] + list(candidates)
    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p
    raise SystemExit(
        f"no data file for task {task['name']!r}. Looked at: "
        + ", ".join(str(Path(c).expanduser()) for c in candidates)
        + "\nPass --data <file.jsonl>.")


def load_items(task: dict, data: Path, n: int | None) -> list[dict]:
    items = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    return items[:n] if n else items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def normalize(value: str | None, how: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if how == "number":
        # Thousands separators, a trailing period, and a stray currency symbol are
        # formatting, not disagreement.
        text = text.replace(",", "").replace("$", "").rstrip(".").strip()
        # "42.0" and "42" are the same answer.
        if re.fullmatch(r"-?\d+\.0+", text):
            text = text.split(".")[0]
    elif how == "lower":
        text = text.lower()
    return text or None


def extract(text: str | None, pattern: str | None) -> str | None:
    """The LAST match, because a chain of thought restates numbers before concluding."""
    if text is None:
        return None
    if not pattern:
        return text.strip()
    found = re.findall(pattern, text, flags=re.MULTILINE)
    return found[-1] if found else None


def score_one(task: dict, generated: str | None, gold_raw: str) -> dict:
    gold = normalize(
        extract(gold_raw, task.get("gold_extract")), task.get("normalize", "none"))
    pred = normalize(
        extract(generated, task.get("pred_extract")), task.get("normalize", "none"))
    marker = task.get("answer_marker")
    return {
        "pred": pred,
        "gold": gold,
        "ok": bool(pred is not None and pred == gold),
        # The run produced nothing for this item — it crashed, timed out, or was never
        # asked. Distinct from `unmarked`: absence of a run is not truncation of one, and
        # conflating them makes a half-finished arm look like a budget problem.
        "missing": generated is None,
        # A marker the model was asked for and did not emit means it never got to the
        # answer — almost always the budget running out mid-reasoning. Counted separately
        # because it is a protocol failure, not a wrong answer.
        "unmarked": bool(marker and generated is not None and marker not in generated),
    }


def score_run(task: dict, items: list[dict], generations: dict[int, str]) -> dict:
    rows, correct, unmarked, missing = [], 0, 0, 0
    for i, item in enumerate(items):
        row = score_one(task, generations.get(i), str(item[task["gold_field"]]))
        row["i"] = i
        correct += row["ok"]
        unmarked += row["unmarked"]
        missing += row["missing"]
        rows.append(row)
    n = len(items)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "unmarked": unmarked,
        "unmarked_rate": unmarked / n if n else 0.0,
        "missing": missing,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# The arm: everything that decides whether two numbers may be subtracted
# ---------------------------------------------------------------------------

# Differ on any of these and the comparison is measuring the difference, not the models.
# `max_new_tokens` is on the list because it is what produced this project's published
# wrong answer; `template_digest` because whether a thinking model thinks by default is a
# property of the template renderer, not of the weights.
PROTOCOL_FIELDS = [
    "task",
    "n",
    "data_digest",
    "instruction_digest",
    "template_digest",
    "max_new_tokens",
    "temperature",
    "stop",
]

# These are the things a comparison is FOR. Never required to match.
FREE_FIELDS = ["arm", "bundle", "driver", "device", "os", "precision", "notes"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def make_arm(task: dict, items: list[dict], args, extra: dict) -> dict:
    body = "\n".join(str(item[task["input_field"]]) for item in items)
    arm = {
        "task": task["name"],
        "n": len(items),
        "data_digest": digest(body),
        "instruction_digest": digest(task.get("instruction", "")),
        "template_digest": args.template_digest or "unrecorded",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        # Not "default": two sloppy runs would then agree on a word that means nothing and
        # compare as if matched. Unrecorded has to look unrecorded on both sides.
        "stop": args.stop or "unrecorded",
        "arm": args.arm or "unnamed",
        "device": args.device or platform.machine(),
        "os": platform.platform(),
    }
    arm.update({k: v for k, v in extra.items() if v is not None})
    return arm


def unrecorded_protocol(arm: dict) -> list[str]:
    """Protocol fields nobody filled in. Unrecorded is not 'matching' — it is unknown."""
    return [f for f in PROTOCOL_FIELDS
            if arm.get(f) in (None, "unrecorded", "")]


# ---------------------------------------------------------------------------
# Comparison — the part that refuses
# ---------------------------------------------------------------------------


def compare(a: dict, b: dict) -> tuple[int, list[str]]:
    """(exit code, lines). Refuses to print a delta unless the protocols match."""
    arm_a, arm_b = a["arm"], b["arm"]
    lines = [
        f"A  {arm_a['arm']:<28} {a['score']['correct']}/{a['score']['n']} "
        f"({a['score']['accuracy']:.1%})   unmarked {a['score']['unmarked']}",
        f"B  {arm_b['arm']:<28} {b['score']['correct']}/{b['score']['n']} "
        f"({b['score']['accuracy']:.1%})   unmarked {b['score']['unmarked']}",
        "",
    ]

    mismatched = [f for f in PROTOCOL_FIELDS if arm_a.get(f) != arm_b.get(f)]
    if mismatched:
        lines.append("REFUSED — the arms were not run under the same protocol:")
        for f in mismatched:
            lines.append(f"    {f:<20} A={arm_a.get(f)!r}   B={arm_b.get(f)!r}")
        lines += [
            "",
            "    The two numbers above are real; the difference between them is not "
            "attributable",
            "    to the models until these agree. Re-run the shorter arm with the other's "
            "settings.",
        ]
        if "max_new_tokens" in mismatched:
            lines.append(
                "    Note max_new_tokens: this is the field that produced a published "
                "12-point")
            lines.append(
                "    'quality gap' in this project which was entirely a 600-vs-2048 "
                "budget artifact.")
        return 3, lines

    unknown = sorted(set(unrecorded_protocol(arm_a)) | set(unrecorded_protocol(arm_b)))
    if unknown:
        lines += [
            "REFUSED — the protocol is not fully recorded, so it cannot be shown to match:",
            "    " + ", ".join(unknown),
            "",
            "    Unrecorded is not the same as equal. Fill these in (--template-digest, "
            "--stop, …)",
            "    or re-run through this harness so they are captured.",
        ]
        return 3, lines

    incomplete = [(n, s["score"].get("missing", 0)) for n, s in (("A", a), ("B", b))
                  if s["score"].get("missing", 0)]
    if incomplete:
        lines.append("REFUSED — an arm did not produce a generation for every item:")
        for name, count in incomplete:
            lines.append(f"    {name} is missing {count} of "
                         f"{(a if name == 'A' else b)['score']['n']}")
        lines += [
            "",
            "    Those items are scored wrong, so the accuracy above is a floor, not a "
            "measurement.",
            "    Re-run the missing items, or score both arms over the items they share.",
        ]
        return 3, lines

    delta = b["score"]["accuracy"] - a["score"]["accuracy"]
    lines.append(f"protocol matched on: {', '.join(PROTOCOL_FIELDS)}")
    lines.append(f"delta  B - A = {delta:+.1%} "
                 f"({b['score']['correct'] - a['score']['correct']:+d} items)")

    # Equal budgets do not mean equal truncation. An arm that runs out of room more often is
    # being scored on a harder task, and the delta is partly measuring that.
    ua, ub = a["score"]["unmarked_rate"], b["score"]["unmarked_rate"]
    if abs(ua - ub) >= 0.05:
        lines += [
            "",
            f"WARNING — truncation differs: A {ua:.0%} of items produced no answer marker, "
            f"B {ub:.0%}.",
            "    The budgets match, so one arm is spending more of it before answering. "
            "Part of",
            "    this delta is that, not quality. Raise --max-new-tokens for both and re-run.",
        ]
    elif max(ua, ub) >= 0.05:
        lines += [
            "",
            f"NOTE — both arms leave {max(ua, ub):.0%} of items unanswered at this budget. "
            "The delta",
            "    is comparable, but both numbers are below what these models can do.",
        ]

    # Where they actually differ, so the next question ("is it the same items?") is one
    # command away rather than a fresh script.
    flips = [r["i"] for ra, rb in zip(a["score"]["rows"], b["score"]["rows"])
             for r in [ra] if ra["ok"] != rb["ok"]]
    if flips:
        lines.append(f"disagree on {len(flips)} items: "
                     + ", ".join(str(i) for i in flips[:20])
                     + (" …" if len(flips) > 20 else ""))
    return 0, lines


# ---------------------------------------------------------------------------
# Generations in
# ---------------------------------------------------------------------------


def read_generations(path: Path) -> dict[int, str]:
    """Whatever a driver dumped, as {item index: generated text}.

    Three shapes are accepted because three already exist in this project: a plain list, a
    dict keyed by index, and the device batch format ({"id": …, "text"/"output": …}).
    Insisting on one would have meant rewriting the device app.
    """
    raw = json.loads(path.read_text())
    out: dict[int, str] = {}
    if isinstance(raw, list):
        for i, entry in enumerate(raw):
            if isinstance(entry, str):
                out[i] = entry
            elif isinstance(entry, dict):
                key = entry.get("id", entry.get("i", i))
                out[int(key)] = _text_of(entry, path)
    elif isinstance(raw, dict):
        body = raw.get("generations", raw)
        for key, entry in body.items():
            out[int(key)] = entry if isinstance(entry, str) else _text_of(entry, path)
    else:
        raise SystemExit(f"{path}: expected a list or an object of generations")
    return out


def _text_of(entry: dict, path: Path) -> str:
    for field in ("text", "output", "generated", "completion", "response"):
        if field in entry:
            return str(entry[field])
    raise SystemExit(
        f"{path}: an entry has no text field (looked for text/output/generated/"
        f"completion/response). Token ids alone cannot be scored here — decode them in "
        f"the driver, where the tokenizer already is.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def cmd_tasks() -> int:
    for name, task in BUILTIN_TASKS.items():
        print(f"{name:<12} {task['description']}")
        for c in task.get("data_candidates", []):
            mark = "✓" if Path(c).expanduser().exists() else " "
            print(f"   {mark} {c}")
    print("\nA task file takes the same shape — see BUILTIN_TASKS in this file.")
    return 0


def cmd_score(args) -> int:
    task = load_task(args.task)
    data = resolve_data(task, args.data)
    items = load_items(task, data, args.n)
    generations = read_generations(Path(args.score).expanduser())

    missing = [i for i in range(len(items)) if i not in generations]
    if missing:
        print(f"note: {len(missing)} of {len(items)} items have no generation "
              f"(scored as wrong): {missing[:10]}", file=sys.stderr)

    score = score_run(task, items, generations)
    arm = make_arm(task, items, args, {
        "bundle": args.bundle,
        "driver": args.driver,
        "precision": args.precision,
        "notes": args.notes,
    })
    transcript = {"arm": arm, "score": score, "data": str(data)}

    print(f"{arm['arm']}: {score['correct']}/{score['n']} ({score['accuracy']:.1%})")
    if score["unmarked"]:
        print(f"  {score['unmarked']} of {score['n']} ({score['unmarked_rate']:.0%}) "
              f"produced no {task.get('answer_marker')!r} — truncated or off-format, "
              f"not merely wrong")
    unknown = unrecorded_protocol(arm)
    if unknown:
        print(f"  protocol unrecorded: {', '.join(unknown)} — this transcript cannot be "
              f"compared until they are filled in")

    if args.transcript:
        Path(args.transcript).write_text(json.dumps(transcript, indent=1))
        print(f"  wrote {args.transcript}")
    return 0


def cmd_compare(args) -> int:
    a = json.loads(Path(args.compare[0]).expanduser().read_text())
    b = json.loads(Path(args.compare[1]).expanduser().read_text())
    code, lines = compare(a, b)
    print("\n".join(lines))
    return code


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task accuracy for a port, and the protocol discipline to compare two.")
    p.add_argument("--tasks", action="store_true", help="list the built-in tasks")
    p.add_argument("--score", metavar="GEN.JSON",
                   help="score a file of generations produced by any driver")
    p.add_argument("--compare", nargs=2, metavar=("A.JSON", "B.JSON"),
                   help="compare two transcripts, refusing on a protocol mismatch")
    p.add_argument("--task", default="gsm8k", help="built-in name or path to a task file")
    p.add_argument("--data", help="override the task's data file")
    p.add_argument("-n", type=int, help="score only the first N items")
    p.add_argument("--arm", help="name for this run, e.g. 'iphone17pro int8'")
    p.add_argument("--max-new-tokens", type=int, required=False,
                   help="the generation budget the run used (PROTOCOL — must be recorded)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--stop", help="how generation was stopped, e.g. 'eos,turn_end'")
    p.add_argument("--template-digest",
                   help="identifier for the chat template AS RENDERED (thinking on/off "
                        "differs by renderer, and it changes the answer)")
    p.add_argument("--bundle", help="what was run (free field)")
    p.add_argument("--driver", help="llm-runner / device / transformers (free field)")
    p.add_argument("--device", help="where it ran (free field)")
    p.add_argument("--precision", help="int8 / fp16 / … (free field)")
    p.add_argument("--notes")
    p.add_argument("--transcript", help="write the arm + score here, for --compare")
    args = p.parse_args()

    if args.tasks:
        return cmd_tasks()
    if args.compare:
        return cmd_compare(args)
    if args.score:
        if args.max_new_tokens is None:
            print("--max-new-tokens is required: it is the protocol field that has already "
                  "produced one wrong published conclusion in this project. Record what the "
                  "run actually used.", file=sys.stderr)
            return 2
        return cmd_score(args)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
