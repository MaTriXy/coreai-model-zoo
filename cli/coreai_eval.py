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


def score_one(task: dict, generated: str | None, gold_raw: str,
              capped: bool | None = None) -> dict:
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
        # No marker: the model was asked for one and did not produce it.
        "unmarked": bool(marker and generated is not None and marker not in generated),
        # Whether generation stopped because it ran out of budget. `None` means the driver
        # did not report a token count, which is the usual case for `--score`.
        #
        # This split matters and was missing until a real run exposed it. Qwen3-0.6B answered
        # a GSM8K item correctly-shaped in 162 tokens of a 512 budget and wrote `\boxed{0}`
        # instead of `#### 0`. Unmarked, but not truncated — and the two have opposite fixes:
        # raise the budget, or fix the prompt. Reporting "raise the budget" there is exactly
        # the misattribution this file exists to stop, committed by this file.
        "truncated": capped,
    }


def score_run(task: dict, items: list[dict], generations: dict[int, str],
              capped: dict[int, bool] | None = None) -> dict:
    rows, correct, unmarked, missing, truncated = [], 0, 0, 0, 0
    for i, item in enumerate(items):
        row = score_one(task, generations.get(i), str(item[task["gold_field"]]),
                        (capped or {}).get(i))
        row["i"] = i
        correct += row["ok"]
        unmarked += row["unmarked"]
        missing += row["missing"]
        truncated += bool(row["truncated"])
        rows.append(row)
    n = len(items)
    # Unmarked AND not truncated: the model finished and ignored the requested format. A
    # different problem from running out of room, and a different fix.
    offformat = sum(1 for r in rows if r["unmarked"] and r["truncated"] is False)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "unmarked": unmarked,
        "unmarked_rate": unmarked / n if n else 0.0,
        "truncated": truncated,
        "truncated_rate": truncated / n if n else 0.0,
        "offformat": offformat,
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
    ta = a["score"].get("truncated_rate", 0.0)
    tb = b["score"].get("truncated_rate", 0.0)
    if abs(ta - tb) >= 0.05:
        lines += [
            "",
            f"WARNING — truncation differs: A ran out of budget on {ta:.0%} of items, "
            f"B on {tb:.0%}.",
            "    The budgets match, so one arm is spending more of it before answering. "
            "Part of",
            "    this delta is that, not quality. Raise --max-new-tokens for both and re-run.",
        ]
    elif max(ta, tb) >= 0.05:
        lines += [
            "",
            f"NOTE — both arms run out of budget on {max(ta, tb):.0%} of items. The delta is",
            "    comparable, but both numbers are below what these models can do.",
        ]

    # Off-format is a different fault with a different fix, and if it is large the task is
    # not measuring what it claims to on either arm.
    oa, ob = a["score"].get("offformat", 0), b["score"].get("offformat", 0)
    if max(oa, ob) >= 0.1 * a["score"]["n"]:
        lines += [
            "",
            f"NOTE — {oa} of A and {ob} of B finished without the answer marker and were "
            f"not truncated.",
            "    That is the model ignoring the requested format, not running out of room. "
            "Raising",
            "    the budget will not move it; the instruction or the extraction pattern is "
            "what to fix.",
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



# ---------------------------------------------------------------------------
# Running a bundle (the integrated path)
# ---------------------------------------------------------------------------

# Rendering and tokenising need the model's own chat template, which lives in
# `transformers` and not in this file. It runs in the same out-of-process shape
# `coreai_verify.ORACLE_SRC` uses, for the same reason: this stays stdlib, and the heavy
# dependency stays optional and visible.
TOKENIZE_SRC = r'''
import json, sys
from transformers import AutoTokenizer

hf_id, revision, thinking, path = sys.argv[1], sys.argv[2] or None, sys.argv[3], sys.argv[4]
prompts = json.load(open(path))
tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)

kw = {}
if thinking in ("on", "off"):
    kw["enable_thinking"] = thinking == "on"

def render(text, tokenize):
    try:
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=tokenize, add_generation_prompt=True, **kw)
    except TypeError:
        # Not every template takes enable_thinking. Falling back silently would make the
        # recorded template digest a lie, so the caller is told which one it got.
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=tokenize, add_generation_prompt=True)

probe = render("PROBE", False)
ids = [render(p, True) for p in prompts]
took_thinking = "enable_thinking" in (tok.chat_template or "")
print("<<<JSON>>>" + json.dumps({
    "probe": probe, "ids": ids, "eos": tok.eos_token_id,
    "template_honours_thinking": took_thinking,
}))
'''

DETOKENIZE_SRC = r'''
import json, sys
from transformers import AutoTokenizer
hf_id, revision, path = sys.argv[1], sys.argv[2] or None, sys.argv[3]
tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
rows = json.load(open(path))
print("<<<JSON>>>" + json.dumps(
    {k: tok.decode(v, skip_special_tokens=True) for k, v in rows.items()}))
'''


def _call_helper(python: str, src: str, args: list[str], payload) -> dict:
    import os
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        data = f.name
    with tempfile.NamedTemporaryFile("w", prefix="coreai_eval_", suffix=".py",
                                     delete=False) as f:
        f.write(src)
        script = f.name
    env = {**os.environ, "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        r = subprocess.run([python, script, *args, data], capture_output=True, text=True,
                           env=env, cwd=tempfile.gettempdir())
    finally:
        Path(script).unlink(missing_ok=True)
        Path(data).unlink(missing_ok=True)
    line = next((x for x in r.stdout.splitlines() if x.startswith("<<<JSON>>>")), None)
    if not line:
        raise SystemExit("tokenizer helper failed:\n" + r.stdout[-600:] + r.stderr[-1200:])
    return json.loads(line[len("<<<JSON>>>"):])


def _hit_budget(tail: str, budget: int) -> bool | None:
    """Did generation stop at the budget, or on its own? llm-runner prints the count in its
    summary — 'Generation: 466.0ms, 162 tokens, …'. Without it, this stays unknown rather
    than guessing, because guessing is what turns a format problem into a budget problem."""
    m = re.search(r"Generation:\s*[\d.]+ms,\s*(\d+)\s*tokens", tail or "")
    return int(m.group(1)) >= budget if m else None


def cmd_run(args) -> int:
    """Drive a bundle over a task and score it, recording the protocol as it goes.

    The recording is the point. Every field `--compare` checks is captured from what
    actually happened — the rendered template, the budget, the stop condition, the driver —
    instead of relying on whoever ran it to type them in afterwards. An arm nobody had to
    describe is an arm nobody can describe wrong.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import coreai_verify as verify  # noqa: E402  (optional: only --run needs it)

    bundle = Path(args.run).expanduser().resolve()
    task = load_task(args.task)
    data = resolve_data(task, args.data)
    items = load_items(task, data, args.n)

    facts = verify.bundle_facts(bundle)
    hf_id = args.hf_id or facts.get("hf_id")
    if not hf_id:
        raise SystemExit(
            f"{bundle}: the bundle does not record its source model, so there is no "
            f"tokenizer and no chat template to render with. Pass --hf-id "
            f"(e.g. --hf-id Qwen/Qwen3-0.6B).")

    runner = verify.resolve_runner(args.runner)
    driver, blockers = verify.driver_plan(facts, runner)
    if blockers and not args.ignore_gpu_lock:
        for b in blockers:
            print(f"blocked: {b}", file=sys.stderr)
        return 4
    print(f"driver {driver}  ·  {len(items)} items  ·  budget {args.max_new_tokens}",
          file=sys.stderr)

    python = verify.resolve_python(args.python)
    prompts = [str(item[task["input_field"]]) + task.get("instruction", "")
               for item in items]
    tok = _call_helper(python, TOKENIZE_SRC,
                       [hf_id, facts.get("revision") or "", args.thinking], prompts)

    generations: dict[int, str] = {}
    capped: dict[int, bool] = {}
    pending_ids: dict[str, list[int]] = {}
    for i, ids in enumerate(tok["ids"]):
        if driver == "llm-runner":
            text, tail = verify.run_llm_runner(runner, bundle, ids, args.max_new_tokens,
                                               bool(facts.get("static_query")))
            if text is None:
                print(f"  item {i}: driver produced nothing — {tail[:200]}",
                      file=sys.stderr)
            else:
                generations[i] = text
                capped[i] = _hit_budget(tail, args.max_new_tokens)
        else:
            out, tail = verify.run_python_cpu(python, bundle, Path(facts["asset"]), ids,
                                              args.max_new_tokens)
            if out is None:
                print(f"  item {i}: driver produced nothing — {tail[:200]}",
                      file=sys.stderr)
            else:
                pending_ids[str(i)] = out
                capped[i] = len(out) >= args.max_new_tokens
        print(f"  {i + 1}/{len(items)}", end="\r", file=sys.stderr)

    if pending_ids:
        # Decoded in one batch, by the same tokenizer that encoded — never by this file.
        decoded = _call_helper(python, DETOKENIZE_SRC,
                               [hf_id, facts.get("revision") or ""], pending_ids)
        generations.update({int(k): v for k, v in decoded.items()})

    score = score_run(task, items, generations, capped)
    arm = make_arm(task, items, args, {
        "bundle": str(bundle),
        "driver": driver,
        "precision": args.precision,
        "notes": args.notes,
        "hf_id": hf_id,
    })
    # Recorded from what happened, overriding whatever the flags said.
    arm["template_digest"] = digest(tok["probe"]) + f"/thinking={args.thinking}"
    arm["stop"] = f"eos={tok['eos']},budget"
    transcript = {"arm": arm, "score": score, "data": str(data),
                  "template_probe": tok["probe"]}

    print()
    print(f"{arm['arm']}: {score['correct']}/{score['n']} ({score['accuracy']:.1%})")
    if score["missing"]:
        print(f"  {score['missing']} items produced nothing — the run is incomplete and "
              f"this number is a floor")
    if score["truncated"]:
        print(f"  {score['truncated']} of {score['n']} ran out of budget mid-answer")
    if score["offformat"]:
        print(f"  {score['offformat']} of {score['n']} finished without "
              f"{task.get('answer_marker')!r} — the model is not following the requested "
              f"format, which raising the budget will not fix")
    if args.transcript:
        Path(args.transcript).write_text(json.dumps(transcript, indent=1))
        print(f"  wrote {args.transcript}")
    return 0


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
    p.add_argument("--run", metavar="BUNDLE",
                   help="drive this bundle over the task, then score it")
    p.add_argument("--hf-id", help="source model, for the tokenizer and chat template")
    p.add_argument("--thinking", choices=["on", "off", "default"], default="default",
                   help="enable_thinking on the chat template — it changes the answer, so "
                        "it is part of the recorded protocol")
    p.add_argument("--runner", help="path to llm-runner")
    p.add_argument("--python", help="python with transformers, for tokenizing")
    p.add_argument("--ignore-gpu-lock", action="store_true")
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
    if args.run:
        if args.max_new_tokens is None:
            print("--max-new-tokens is required: the generation budget is a protocol field "
                  "and there is no safe default.", file=sys.stderr)
            return 2
        return cmd_run(args)
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
