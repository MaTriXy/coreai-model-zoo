"""Oracle dump for Ornith-1.0-9B (qwen3_5 dense hybrid, untied head).

fp32 eager CPU forward of the HF ConditionalGeneration checkpoint (text-only
input; the VL wrapper routes to the language model). One teacher-forced forward
over the fixed prompt gives per-position top-1 + top-2 margins; then a greedy
continuation (full re-forwards, no cache) seeds the decode check.

Margin rule: every gated position must have top-2 margin >= 0.1 (validated
here, BEFORE any bundle exists). Prompt = the code prompt proven on the qwen3.5
family (verify-chunk parity gate) — code text keeps greedy margins fat.

Run:  ~/.venv-coreml-llm-py312/bin/python _smoke/gen_ornith9b_ref.py
Out:  _smoke/ornith9b_ref.pt
"""
from __future__ import annotations

import os

import torch

HF_ID = os.environ.get("ORNITH_HF_ID", "deepreinforce-ai/Ornith-1.0-9B")
OUT = os.environ.get("ORNITH_REF", os.path.join(os.path.dirname(__file__), "ornith9b_ref.pt"))
N_SWEEP = 16     # gated teacher-forced positions (the last N of the prompt)
N_GREEDY = 12    # greedy continuation length

PROMPT = ("def fib(n):\n    if n < 2:\n        return n\n"
          "    return fib(n - 1) + fib(n - 2)\n\nprint(fib(10))")


@torch.no_grad()
def main() -> None:
    from transformers import AutoTokenizer

    try:
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForImageTextToText as Cls

    tok = AutoTokenizer.from_pretrained(HF_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    n = ids.shape[1]
    print(f"prompt tokens: {n} (sweep = last {N_SWEEP} predictions)")
    assert n - 1 >= N_SWEEP, "prompt too short for the sweep"

    print(f"loading {HF_ID} fp32 eager CPU ...", flush=True)
    model = Cls.from_pretrained(HF_ID, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    logits = model(input_ids=ids).logits.float()  # [1, n, V]

    top2 = logits[0].topk(2, dim=-1)
    top1 = top2.indices[:, 0].clone()             # position i predicts token i+1
    margin = (top2.values[:, 0] - top2.values[:, 1]).clone()

    sweep = list(range(n - 1 - N_SWEEP, n - 1))
    print("pos | next(actual) | oracle top1 | margin")
    ok = True
    for p in sweep:
        nxt = int(ids[0, p + 1])
        t1 = int(top1[p])
        m = float(margin[p])
        flag = "" if m >= 0.1 else "  <-- SUB-MARGIN"
        ok &= m >= 0.1
        print(f"{p:3d} | {nxt:6d} | {t1:6d} | {m:8.4f}{flag}")

    greedy = ids.clone()
    for _ in range(N_GREEDY):
        lg = model(input_ids=greedy).logits[0, -1]
        greedy = torch.cat([greedy, lg.argmax().view(1, 1).to(greedy.dtype)], dim=1)
    greedy_ids = greedy[0, n:].tolist()
    # margins along the greedy continuation (for the decode check's margin rule)
    greedy_margin = []
    for i in range(N_GREEDY):
        lg = model(input_ids=greedy[:, : n + i]).logits[0, -1].float()
        t2 = lg.topk(2)
        greedy_margin.append(float(t2.values[0] - t2.values[1]))
    print(f"greedy continuation: {greedy_ids}")
    print(f"greedy text: {tok.decode(greedy_ids)!r}")
    print(f"greedy margins: {[round(m, 3) for m in greedy_margin]}")

    torch.save(
        {
            "hf_id": HF_ID,
            "prompt": PROMPT,
            "ids": ids,
            "top1": top1,
            "margin": margin,
            "sweep_positions": sweep,
            "greedy_ids": greedy_ids,
            "greedy_margin": greedy_margin,
        },
        OUT,
    )
    print(f"saved {OUT}")
    print(f"MARGIN GATE (all sweep >= 0.1): {'PASS' if ok else 'FAIL — pick another prompt'}")


if __name__ == "__main__":
    main()
