#!/usr/bin/env python3
"""
Pre-tokenize REAL code / RAG prompts with the bundle's own BPE tokenizer so the
on-device spec-decode probe (PB_PROMPT_IDS) measures REAL per-domain alpha,
not the synthetic repetitive-token ceiling.

Emits comma-separated int32 id lists (one per prompt) + token counts. Feed a
list to PipelinedBench via PB_PROMPT_IDS.

Usage:
  specdecode_pretokenize_realprompt.py <tokenizer_dir> [out_dir]
  tokenizer_dir = e.g. coreai-models/exports/qwen3_8b_gpu/tokenizer
"""
import sys
import os
from transformers import AutoTokenizer

# --- Prompt 1: CODE completion (raw, no chat template). A real code prefix; the
# model continues it, reusing identifiers/patterns already in scope -> n-gram hits.
CODE_PROMPT = '''// Greedy speculative decode (fused scheme): each round forwards
// [a0, draft_0 ... draft_{K-1}] where a0 = argmax(cachedLogits) is the guaranteed
// next greedy token. Rollback is free: advance `processed` past accepted tokens only.
func specDecodeRound(prompt: [Int32], K: Int, drafter: PromptLookupDrafter) -> [Int32] {
    var processed = 0
    var seq = prompt
    var out: [Int32] = []
    while out.count < gen {
        let a0 = argmaxRow(cached, row: 0, vocab: vocab)
        let draft = drafter.draft(seq + [a0], K)
        let verifyTokens = [a0] + draft
        let flat = try await driver.forward(verifyTokens, processed: processed, &s0, &s1)
        var accepted = 0
        for i in 0..<draft.count {
            let want = argmaxRow(flat, row: i, vocab: vocab)
            if draft[i] == want { accepted += 1 } else { break }
        }
        let committed = [a0] + Array(draft[0..<accepted])
        for t in committed { seq.append(t); out.append(t) }
        processed += committed.count
        cached = Array(flat[(accepted * vocab)..<((accepted + 1) * vocab)])
    }
    return out
}

// Continue: add a helper that rolls the KV cache back to an accepted length and
'''

# --- Prompt 2: RAG (chat template, thinking off). A document + a question whose
# answer must quote the document -> the grounded answer echoes context n-grams.
RAG_DOC = (
    "Speculative decoding accelerates autoregressive generation by drafting several "
    "candidate tokens cheaply and verifying them with a single forward pass of the "
    "target model. In the greedy variant the output is token-identical to plain greedy "
    "decoding, so it is lossless. Prompt-lookup drafting requires no draft model: it "
    "proposes the continuation that most recently followed the current n-gram in the "
    "context. Its acceptance rate is high on code, JSON, and retrieval-augmented "
    "answers because those repeat their own n-grams, and low on free-form prose. The "
    "verify forward runs the K draft tokens at positions n..n+K-1 in one pass and reads "
    "the argmax at every position; rejected draft KV rows are simply overwritten by the "
    "next forward and never attended, so rollback costs nothing."
)
RAG_QUESTION = (
    "Based on the passage, explain why prompt-lookup speculative decoding is lossless, "
    "why it needs no draft model, and on what kind of text its acceptance rate is high."
)


def flatten_ids(x):
    # Normalize AutoTokenizer / apply_chat_template returns (list, BatchEncoding dict,
    # or batch-nested list) to a flat list[int].
    if not isinstance(x, (list, tuple)):   # dict / BatchEncoding(UserDict) / tensor
        x = x["input_ids"]
    while len(x) and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [int(v) for v in x]


def main():
    tokdir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    tok = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=False)

    prompts = {}
    prompts["code"] = flatten_ids(tok(CODE_PROMPT, add_special_tokens=False)["input_ids"])

    messages = [
        {"role": "system", "content": "Answer strictly using the passage."},
        {"role": "user", "content": f"Passage:\n{RAG_DOC}\n\nQuestion: {RAG_QUESTION}"},
    ]
    try:
        rag_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except Exception as e:
        print(f"  (chat_template enable_thinking unsupported: {e}; retrying without)")
        rag_ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    prompts["rag"] = flatten_ids(rag_ids)

    for name, ids in prompts.items():
        csv = ",".join(str(x) for x in ids)
        path = os.path.join(out_dir, f"specprompt_{name}.ids")
        with open(path, "w") as fh:
            fh.write(csv)
        print(f"[{name}] {len(ids)} tokens -> {path}")
        print(f"    head: {ids[:12]}")
    print("\nfeed to probe:  PB_PROMPT_IDS=$(cat specprompt_code.ids) ...")


if __name__ == "__main__":
    main()
