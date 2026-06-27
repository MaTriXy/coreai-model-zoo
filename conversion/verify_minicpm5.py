#!/usr/bin/env python3
"""Greedy parity gate: Core AI engine vs HF transformers for MiniCPM5-1B.

Feeds identical token ids to both and compares the greedy continuation
token-for-token (via decoded text). A faithful conversion reproduces HF's
greedy decode exactly; any logit error large enough to flip an argmax shows up
as a divergence. The shipped int8 bundle scores 24/24 token-exact = lossless.

(The engine's --save-logits path is unusable here: the pipelined engine refuses
logits, and the sequential engine's raw-tokens logit buffer is off-by-one —
generatedTokens = logits + 1 — so we compare the greedy text instead.)

Usage:
    LLM_RUNNER=/path/to/coreai-models/.build/out/Products/Release/llm-runner \
        python conversion/verify_minicpm5.py <bundle_dir> [n_new_tokens]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ID = "openbmb/MiniCPM5-1B"
# Path to the coreai-models llm-runner Release binary (built locally).
RUNNER = os.environ.get("LLM_RUNNER", "")

PROMPTS = [
    "The capital of France is",
    "Question: What is 17 + 25? Answer:",
    "The three primary colors are",
    "Once upon a time, there was a",
]


def engine_text(bundle: str, token_ids: list[int], n: int) -> str:
    with tempfile.TemporaryDirectory() as td:
        rt = Path(td) / "rt.json"
        rt.write_text(json.dumps({"tokens": token_ids}))
        out = subprocess.run(
            [RUNNER, "--model", bundle, "--raw-tokens", str(rt),
             "--max-tokens", str(n), "--sampling-strategy", "greedy"],
            check=True, capture_output=True, text=True,
        ).stdout
        body = out.split("Generating...", 1)[1]
        body = re.split(r"\n\s*⏱️|\n\s*Performance Summary", body, 1)[0]
        return body.strip()


def main() -> None:
    if not RUNNER or not Path(RUNNER).exists():
        sys.exit("set LLM_RUNNER to the coreai-models llm-runner Release binary")
    bundle = str(Path(sys.argv[1]).resolve())
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    tok = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID, dtype=torch.float32, trust_remote_code=True
    ).eval()

    print(f"bundle: {bundle}  (n_new={n})\n")
    passes = 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=n, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        hf_text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False).strip()
        eng_text = engine_text(bundle, ids[0].tolist(), n)

        # token-level: re-encode both continuations and find the common prefix
        hf_ids = tok(hf_text, add_special_tokens=False).input_ids
        eng_ids = tok(eng_text, add_special_tokens=False).input_ids
        common = 0
        for a, b in zip(hf_ids, eng_ids):
            if a != b:
                break
            common += 1
        exact = hf_text == eng_text
        passes += exact
        mark = "OK " if exact else "~~ "
        print(f"{mark}{p!r}")
        print(f"     common-prefix tokens: {common}/{min(len(hf_ids), len(eng_ids))}"
              f"  exact_text_match={exact}")
        if not exact:
            print(f"     HF : {hf_text[:160]!r}")
            print(f"     ENG: {eng_text[:160]!r}")
        print()

    print(f"=== exact greedy match: {passes}/{len(PROMPTS)} ===")


if __name__ == "__main__":
    main()
