# Qwen3-Reranker-0.6B — Core AI

The **cross-encoder reranker** that closes the on-device RAG loop: **embed → rerank → generate**,
all local. Pairs with [`qwen3-embedding.md`](../qwen3-embedding/README.md) — the embedder does fast
first-stage retrieval, this reranker re-scores the shortlist for precision.
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (Apache-2.0) runs as
a single static `.aimodel`.

**Same Qwen3-0.6B backbone as the embedder, different head.** A cross-encoder reads one
`query + document` sequence and asks the LM a yes/no question — the relevance score is the
softmax weight on the **"yes"** token vs the **"no"** token at the final position. So unlike the
embedder it keeps the **LM head**; like the embedder it is a plain `.aimodel` via `AIModel.run`
(one forward, no KV loop, no generation). The whole scoring tail — gather the last real token,
apply the head to **that one position**, select {no, yes}, softmax — is baked into the graph.

## Graph contract

```
input  "input_ids"       [1, S]  int32   right-padded to grid S (pad id 151643)
input  "attention_mask"  [1, S]  int32   1 over real tokens, 0 over padding
output "probs"           [1, 2]  fp16    softmax([no_logit, yes_logit]); relevance = probs[0,1] = P(yes)
```

Shipped grid **S = 512** (covers query + a ~450-token document; `--seq-len` for other grids).

**Host recipe** (format the pair exactly like the upstream model card, then right-pad):
```
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
body   = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"      # default instruction below
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
input_ids = encode(prefix) + encode(body) + encode(suffix)        # add_special_tokens=False, right-pad to S
```
Default instruction: `Given a web search query, retrieve relevant passages that answer the query`
(swap it per task — the model is instruction-aware). Score each candidate document, sort
descending by `probs[0,1]`. Last-token gather under the causal mask is right-pad safe, so the
host just right-pads (the graph reads the true last token from the mask) — equivalent to the
upstream left-pad + `logits[:, -1]`.

## Measured (macOS 27 beta, M4 Max GPU, fp16, warm)

**45.7 ms / pair-score** at the 512 grid (~22 scores/s), cold load ~0.3–1.9 s. Same cost class as
the embedder (same backbone + grid; the head on a single position is free). Latency scales ~linearly
with the grid — use the smallest S that covers your `query + document`. fp16 is the ship dtype.

## Numerics gate

**Torch ladder** (`conversion/export_qwen3_reranker.py`, vs the official `AutoModelForCausalLM`
scoring, fp32): the in-graph wrapper reproduces the official P(yes) **exactly** (|Δ| = 0.00000
across 6 pairs); relevant pairs score 0.98–1.00, irrelevant ≈ 0.0000; every relevant pair
outranks the irrelevant document sharing its query. Confirms right-pad + mask-gather ≡ the
upstream left-pad + position −1.

**Engine gate** (`_smoke/gate_qwen3rerank_engine.py`, `.aimodel` on the GPU delegate vs the torch
reference): end-to-end (host format → right-pad → `AIModel.run`) P(yes) within **|Δ| < 0.0005**,
ranking identical, fp16 no overflow.

## ⬇️ Bundle

**[mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI)**
— `qwen3-reranker-0.6b_float16_s512_static.aimodel` (~1.1 GB) + `reference.json` (pairs, scores,
prompt scaffolding) + `tokenizer/`. Apache-2.0.

Convert yourself: [`conversion/export_qwen3_reranker.py`](../../conversion/export_qwen3_reranker.py)
(`uv run conversion/export_qwen3_reranker.py --dtype float16 --seq-len 512 --output-dir out`).

## CoreAIKit (Swift)

Runs out of the box with [CoreAIKit](https://github.com/john-rocky/coreai-kit) — it downloads this
repo on first use and formats the pair (prefix + instruct/query/document + suffix) in-process:

```swift
import CoreAIKitEmbeddings

let reranker = try await Reranker(model: .qwen3Reranker0_6B)
let ranked = try await reranker.rerank(
    query: "What is the capital of Japan?",
    documents: ["Tokyo is the capital of Japan.", "Python is a programming language."])
// ranked[0].document is most relevant; ranked[i].score is P(yes) in [0, 1]
```

Pair it with `TextEmbedder(model: .qwen3Embedding0_6B)` for the full retrieve→rerank stack.

## The port in one lesson: a cross-encoder is an LM head on one token

The reranker is a causal LM scored, not generated: read the next-token logits at the last real
token, compare the `yes` (id 9693) and `no` (id 2152) tokens, softmax. Baking that tail
into the graph (gather last hidden → head on that **one** position → 2-way softmax) makes the
output a single `[1, 2]` probability and keeps the head cost negligible. Two dtype traps carry
over from the embedder verbatim: **load fp32** (the checkpoint is bf16) for a clean reference, and
emit fp16 with **`module.half()`, not `autocast`** — autocast-fp16 collides with Qwen3's RMSNorm
fp32 roundtrip (`_assert_tensor_metadata`). Everything else traced clean on the first try.
