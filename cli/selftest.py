#!/usr/bin/env python3
"""Self-test for the decision rules: doctor's source lint and verify's verdict.

The negative half matters more than the positive half. A lint that flags the documented
WORKAROUND as if it were the bug is worse than no lint — it teaches people to ignore it.
The same goes for the gate: a verdict that calls an fp16 knife-edge tie a failure trains
people to override it, and then it catches nothing.

    python3 selftest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_doctor as doctor  # noqa: E402

# (rule id, line that must fire, line that must NOT fire)
CASES: list[tuple[str, str, str]] = [
    ("SRC-CAST-ROUNDTRIP",
     "y = (x + 64.0).long().float() - 64.0",
     "y = torch.div(x * 2.0, 2.0, rounding_mode='floor')"),
    ("SRC-FLOOR-ON-GPU",
     "y = torch.floor(x)",
     "y = torch.div(x * 2.0, 2.0, rounding_mode='floor')"),
    ("SRC-FLOORDIV-ONE",
     'y = torch.div(x, 1, rounding_mode="floor")',
     'y = torch.div(x * 2.0, 2.0, rounding_mode="floor")'),
    ("SRC-INT64-BOOL-MASK",
     "mask = ((ix0 >= 0) & (ix0 < W)).to(dtype)",
     "mask = 1 - (x - x.clamp(0, W)).abs().clamp(max=1)"),
    ("SRC-ARANGE-FLOAT",
     "t = torch.arange(8.0, dtype=x.dtype)",
     "t = torch.arange(8, dtype=x.dtype)"),
    ("SRC-FP16-DECOMP-OVERFLOW",
     "y = F.softplus(x)",
     "y = torch.clamp(x, min=0) + torch.log1p(torch.exp(-x.abs()))"),
    ("SRC-OPTIMIZE-AXIS-MOVE",
     "s2 = torch.sum(y ** 2, dim=-1).unsqueeze(-2)",
     "s2 = torch.sum(y ** 2, dim=-1).reshape(1, 1, -1)"),
    ("SRC-SQUEEZE-DIM",
     "x = x.squeeze(1)",
     "x = x.squeeze()"),
    ("SRC-COMPLEX-OPS",
     "f = torch.polar(mag, ang)",
     "f = torch.stack([cos, sin], -1)"),
    ("SRC-REMAINDER",
     "i = torch.remainder(pos, W)",
     "i = torch.where(pos >= W, pos - W, pos)"),
    ("SRC-F-NORMALIZE",
     "q = F.normalize(q, dim=-1)",
     "q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)"),
    ("SRC-TORCH-ASSERT",
     "torch._assert(n > 0, 'positive')",
     "assert isinstance(n, int)"),
    ("SRC-WHILE-LOOP",
     "out = torch.ops.higher_order.while_loop(cond, body, carry)",
     "out = step(carry)  # loop-free single step at S=1"),
    ("SRC-DATA-INDEXED-KV-WRITE",
     "cache = slice_update(cache, col, begin=in_step)",
     "cache = cache * (1 - write_mask) + col * write_mask"),
]


def findings_for(text: str) -> set[str]:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "m.py"
        f.write_text(text + "\n")
        rep = doctor.Report(target=str(f), kind="source")
        doctor.check_source_files([f], rep)
        return {x.rule.id for x in rep.findings}


def main() -> int:
    failures: list[str] = []
    for rule_id, trigger, workaround in CASES:
        if rule_id not in findings_for(trigger):
            failures.append(f"{rule_id}: MISSED its trigger  -> {trigger}")
        if rule_id in findings_for(workaround):
            failures.append(f"{rule_id}: FIRED on the fix    -> {workaround}")

    # Counting rules: one write site is normal, several on one handle is the bug.
    if "SRC-CHAINED-STATE-WRITES" in findings_for("s = update_states(s, new)"):
        failures.append("SRC-CHAINED-STATE-WRITES: fired on a single write site")
    many = "\n".join(f"s{i} = update_states(s{i}, new{i})" for i in range(3))
    if "SRC-CHAINED-STATE-WRITES" not in findings_for(many):
        failures.append("SRC-CHAINED-STATE-WRITES: missed three write sites")

    # Absence rule: slice_update without remove_functionalization.
    if "SRC-MISSING-DEFUNCTIONALIZE" not in findings_for("ep = slice_update(ep, v)"):
        failures.append("SRC-MISSING-DEFUNCTIONALIZE: missed a bare slice_update")
    paired = "ep = slice_update(ep, v)\nep = remove_functionalization(ep)"
    if "SRC-MISSING-DEFUNCTIONALIZE" in findings_for(paired):
        failures.append("SRC-MISSING-DEFUNCTIONALIZE: fired despite remove_functionalization")

    # Comments are documentation, not code.
    if findings_for("# y = torch.floor(x)  -- do not do this"):
        failures.append("a commented-out line was reported")

    covered = {r.id for r, _p, _f in doctor.SRC_RULES}
    tested = {c[0] for c in CASES} | {"SRC-CHAINED-STATE-WRITES", "SRC-MISSING-DEFUNCTIONALIZE"}
    if untested := covered - tested:
        failures.append(f"source rules with no self-test: {', '.join(sorted(untested))}")

    n_verify = check_verify(failures)

    for line in failures:
        print("FAIL  " + line)
    print(f"\n{len(CASES) + 4} checks over {len(covered)} source rules, "
          f"{n_verify} over verify's verdict: {'FAILED' if failures else 'all pass'}")
    return 1 if failures else 0


def oracle(texts: list[str], margins: list[float]) -> dict:
    """A synthetic oracle result: cumulative decodes plus the per-step top-2 margins."""
    prefixes, acc = [], ""
    for t in texts:
        acc += t
        prefixes.append(acc)
    return {"gen_text": acc, "step_prefixes": prefixes, "margins": margins,
            "gen_ids": list(range(len(texts)))}


def check_verify(failures: list[str]) -> int:
    """The verdict rule, on both sides of the margin floor.

    A gate that cannot distinguish 'the conversion is wrong' from 'fp16 broke a tie' is
    not a gate — it either blocks good bundles or waves through broken ones.
    """
    import coreai_verify as verify

    confident = [0.9] * 4
    knife_edge = [0.9, 0.9, 0.004, 0.9]
    cases = [
        # (label, oracle, bundle text, floor, expected verdict)
        ("identical", oracle([" A", " B", " C", " D"], confident), " A B C D", 0.1, "PASS"),
        ("real divergence, high margin", oracle([" A", " B", " C", " D"], confident),
         " A B X D", 0.1, "FAIL"),
        ("knife-edge tie below the floor", oracle([" A", " B", " C", " D"], knife_edge),
         " A B X D", 0.1, "PASS"),
        ("same tie, floor lowered under it", oracle([" A", " B", " C", " D"], knife_edge),
         " A B X D", 0.001, "FAIL"),
        ("diverges at the very first token", oracle([" A", " B"], confident), " X B", 0.1, "FAIL"),
        ("bundle stopped early", oracle([" A", " B", " C"], confident), " A B", 0.1, "PASS"),
    ]
    for label, orc, got, floor, expected in cases:
        result, _line = verify.judge(orc, got, floor)
        if result != expected:
            failures.append(f"verify.judge — {label}: got {result}, expected {expected}")

    # Prompt validation: a position under the floor makes the prompt unusable, in either
    # direction. This is computable from the oracle alone, before any bundle exists.
    if verify.validate_prompt(oracle([" A", " B"], [0.9, 0.9]), 0.1):
        failures.append("verify.validate_prompt rejected a clean prompt")
    weak = verify.validate_prompt(oracle([" A", " B"], [0.9, 0.012]), 0.1)
    if len(weak) != 1:
        failures.append(f"verify.validate_prompt missed a 0.012-margin tie (got {weak})")
    return len(cases) + 2


if __name__ == "__main__":
    raise SystemExit(main())
