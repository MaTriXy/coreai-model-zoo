#!/usr/bin/env python3
"""fp32 oracle for Shieldstral-1.0-3B: P(unsafe) over a fixed moderation suite.

Shieldstral is a *classifier by construction*: the answer is the next-token
distribution restricted to {"no", "yes"} at the last prompt token. So the oracle
is one forward per case, and what we store is the last-position logits (and the
two we care about), not a generation.

**transformers git main only.** The 4.57.6 release cannot load this checkpoint at
all: its tokenizer declares `TokenizersBackend`, `AutoModelForCausalLM` rejects
`Mistral3Config`, and the zoo's config shim registers `Ministral3TextConfig` with
AutoConfig only. git main knows `ministral3` natively, YARN included.

Run:
    ~/code/litertlm-convert/.venv-vl0930-t515/bin/python _smoke/shieldstral_ref.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import hf_snapshot  # noqa: E402
from _shieldstral_suite import SUITE, SYSTEM  # noqa: E402

HF_ID = "mistralai/Shieldstral-1.0-3B"

def build_prompt(tok, instruction: str, query: str, document: str) -> str:
    user = f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

    print(f"transformers {transformers.__version__}")
    if int(transformers.__version__.split(".")[0]) < 5:
        raise SystemExit("needs transformers >= 5 (git main): 4.x cannot load this checkpoint")

    snap = hf_snapshot(args.hf_id)
    tok = AutoTokenizer.from_pretrained(snap)
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    print(f"yes={yes_id} no={no_id}")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        snap, dtype=torch.float32).eval()

    out: dict[str, np.ndarray] = {}
    print(f"\n{'case':26s} {'len':>4}  P(unsafe)   verdict  expected")
    for i, (instr, query, doc, want_unsafe, label) in enumerate(SUITE):
        ids = tok(build_prompt(tok, instr, query, doc), return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1].float()
        yn = torch.stack([logits[no_id], logits[yes_id]])
        p = float(torch.softmax(yn, 0)[1])
        verdict = p > 0.5
        ok = "OK" if verdict == want_unsafe else "** WRONG SIDE **"
        print(f"{label:26s} {ids.shape[1]:>4}  {p:8.4f}   "
              f"{'UNSAFE' if verdict else 'safe':6s}  {'UNSAFE' if want_unsafe else 'safe':6s} {ok}")
        out[f"case{i}_ids"] = ids[0].numpy().astype(np.int64)
        out[f"case{i}_yn"] = yn.numpy().astype(np.float32)
        out[f"case{i}_p"] = np.float32(p)
        out[f"case{i}_logits_last"] = logits.numpy().astype(np.float32)

    out["_meta_cases"] = np.int32(len(SUITE))
    out["_meta_yes_id"] = np.int32(yes_id)
    out["_meta_no_id"] = np.int32(no_id)
    out["_meta_expected"] = np.array([int(c[3]) for c in SUITE], dtype=np.int32)
    out["_meta_labels"] = np.array([c[4] for c in SUITE])

    dest = Path(args.out) if args.out else Path(__file__).parent / "shieldstral_3b_suite_ref.npz"
    np.savez(dest, **out)
    print(f"\nwrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
