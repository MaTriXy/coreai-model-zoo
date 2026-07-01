#!/usr/bin/env python3
"""
Model-free alpha prior for prompt-lookup (n-gram) speculative decoding on REAL
text. prompt-lookup proposes the continuation that most-recently followed the
current n-gram; its acceptance is bounded by how often real sequences repeat
their own n-grams. We measure that directly on actual files:

  at each position p, take the last `ng` tokens, find the most recent earlier
  occurrence, and count how many of its following tokens match the ACTUAL
  continuation at p (capped at max_draft). This is exactly the drafter's
  accepted-length distribution on a self-consistent stream — a model-free proxy
  for alpha on input-grounded/structured text (code, JSON, RAG) vs prose.

Word-level tokenization (whitespace+punct) approximates subword granularity for
this purpose; absolute numbers shift with a real BPE tokenizer but the
code-vs-prose GAP is the signal.
"""
import re
import sys
import glob


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def measure(tokens, ng=3, max_draft=8):
    n = len(tokens)
    index = {}                       # ngram tuple -> most recent START index seen
    hits = 0
    positions = 0
    accepted_lengths = []
    for p in range(ng, n):
        key = tuple(tokens[p - ng:p])    # current context ngram, starts at p-ng
        positions += 1
        prev = index.get(key)            # a PRIOR start of the same ngram (< p-ng)
        if prev is not None:
            # continuation predicted from prev+ng; count matching tokens vs actual
            acc = 0
            while (acc < max_draft and p + acc < n
                   and tokens[prev + ng + acc] == tokens[p + acc]):
                acc += 1
            if acc > 0:
                hits += 1
                accepted_lengths.append(acc)
        # register AFTER lookup so the current occurrence can't match itself
        index[key] = p - ng
    hit_rate = hits / positions if positions else 0.0
    mean_acc = sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
    # per-position expected accepted draft tokens (0 on misses)
    exp_acc = sum(accepted_lengths) / positions if positions else 0.0
    return hit_rate, mean_acc, exp_acc, positions


def run(label, paths, ng=3):
    text = ""
    used = 0
    for pat in paths:
        for fp in glob.glob(pat):
            try:
                with open(fp, "r", errors="ignore") as fh:
                    text += fh.read() + "\n"
                used += 1
                if len(text) > 400_000:
                    break
            except Exception:
                pass
        if len(text) > 400_000:
            break
    toks = tokenize(text)
    if len(toks) < 100:
        print(f"  {label:26s}  (insufficient text: {len(toks)} tokens, {used} files)")
        return
    hr, ma, ea, pos = measure(toks, ng=ng)
    print(f"  {label:26s}  hit-rate={hr:5.1%}  mean-accept={ma:4.2f}  "
          f"exp-accept/pos={ea:4.2f}  (ng={ng}, {len(toks)} toks)")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    print("prompt-lookup n-gram acceptance on REAL text (model-free proxy)\n")
    print("exp-accept/pos ~ accepted draft tokens per decode step from n-gram alone;")
    print("higher = better spec-decode payoff on that text type.\n")
    for ng in (2, 3):
        print(f"--- n-gram order {ng} ---")
        run("Swift source (code)", [f"{base}/swift/Sources/**/*.swift"], ng=ng)
        run("Python source (code)", [f"{base}/**/*.py"], ng=ng)
        run("Markdown/prose", [f"{base}/**/*.md", f"{base}/knowledge/*.md"], ng=ng)
        print()
