# Youtu-LLM-2B (dense MLA) — Core AI

The **zoo's first Multi-head Latent Attention (MLA) model on iPhone**, and its first
**dense** MLA. [GLM-4.7-Flash](glm-4.7-flash.md) brought the DeepSeek MLA family to the
zoo, but as a 30B **Mac-only MoE**; Youtu-LLM-2B is the same attention at **1.96B dense**,
so the whole model — with a tiny latent KV cache — fits on an iPhone. Source:
[`tencent/Youtu-LLM-2B`](https://huggingface.co/tencent/Youtu-LLM-2B) (`youtu-llm` license,
`model_type youtu`).

Architecturally this is **dense DeepSeek-V2/V3 MLA on every layer + a plain SwiGLU FFN**
(the difference from GLM-4.7-Flash is MoE→dense MLP and a weight-tied head):

- **MLA** (heads 16, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope` 128 + decoupled
  `qk_rope` 64 = head_dim 192, **`v_head_dim` 128** — the asymmetric DeepSeek-V2-Lite
  shape, K≠V — RoPE θ=1.6e6, **interleaved**). Shipped in the **absorbed** form: the KV
  cache holds only the compressed latent `[512]` + the shared rope key `[64]` (2×`[288]`
  halves), and the `kv_b` up-projection is folded into a per-head query lift `W_UK` and a
  latent-value readout `W_UV`. That 576-dim, K(576)≠V(512) MQA shape is exactly what the
  **absorbed-MLA flash-decode Metal kernel** (`mla_metal_sdpa.py`, shared with GLM) computes;
  the naive/materialized form gates the math.
- **Dense SwiGLU FFN** (`intermediate 6144`, silu) on all 32 layers — no MoE, no router.
- **Weight-tied lm_head**, Llama-3 tokenizer (vocab 128256), 128K context, **thinking mode**
  (`<think>…</think>`) + native agentic/tool-use.

**1.96B params, dense** → int8 body + int8 head is ~2.2 GB, well inside the iPhone budget.

**⬇️ Converted `.aimodel` bundle:** `gpu-pipelined/youtu_llm_2b_decode_absorbed_msdpa/` on
[🤗 mlboydaisuke/Youtu-LLM-2B-CoreAI](https://huggingface.co/mlboydaisuke/Youtu-LLM-2B-CoreAI)
(2.2 GB int8, full LanguageBundle incl. tokenizer; decode-only loop-free for the
[pipelined engine](../knowledge/pipelined-engine.md)). Convert with
[`conversion/export_youtu_decode_pipelined.py`](../conversion/export_youtu_decode_pipelined.py).

## Use it

💻 **Build with it** — [CoreAIKit](https://github.com/john-rocky/coreai-kit) (SPM), one line:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "youtu-llm-2b")
let reply = try await chat.respond(to: "What can you do, offline?")
// downloads once, runs fully on-device; the <think> reasoning arrives as .thinking events,
// reply.content is the final answer
```

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(pick "Youtu-LLM 2B"), or the zoo's own [CoreAIChat](../apps/CoreAIChat) on-device app. Verified
on Mac via `swift run chat-cli --model youtu-llm-2b` → *"The capital of France is Paris."*

The bundle is a standard decode-only `LanguageBundle` on Apple's `coreai-pipelined` GPU engine
(`COREAI_CHUNK_THRESHOLD=1`, never `engine.warmup()`), chat-templated
`<|begin_of_text|><|User|>…<|Assistant|>`.

## Measured (release engine, `COREAI_CHUNK_THRESHOLD=1`)

| surface | prefill (S=1) | decode | numerics |
|---|---:|---:|---|
| **M4 Max** (`llm-runner`, greedy, split_g=8) | 95.9 | **102.8 tok/s** | 16/16 top-1 = HF fp32 oracle |
| **iPhone 17 Pro** (PipelinedBench, p=128 g=256) | 20.5 | **~19 tok/s** (in-app warm ~24) | **16/16 · device ≡ Mac ≡ HF** |

engine ready 38.5 s cold on the iPhone (2.2 GB spec); the custom absorbed-MLA Metal kernel
lowers on the A19 GPU (JIT cold-spec, no AOT needed on this path).

## Numerics

- **Authored model == fp32 HF oracle, token-exact.** Naive/materialized: prefill per-position
  cosine **1.000002**, greedy continuation **0 flips / 16**. Absorbed (latent-cache stateful
  decode): cosine 1.000002, **0 flips / 16** — the `W_UK`/`W_UV` factorization + `AbsorbedKVCache`
  are correct. (RoPE interleave convention verified bit-identical to HF `apply_rotary_pos_emb_interleave`.)
- **int8 on the real GPU engine == oracle, byte-for-byte.** On both Mac (`llm-runner`) and iPhone
  (PipelinedBench), greedy of "The capital of France is" reproduces the oracle 16/16:
  `" Paris.Okay, so the user is talking about the capital of France. They"`.

## Notes

- **int8 is the ship floor.** int4-linear (blk32) is 1.5 GB / 123 tok/s on Mac but **forks at
  token 2** vs the oracle (the non-QAT int4 cliff) — dropped. int8 body + int8 head holds
  quality. A future int4-k-means (outlier-robust) is the smaller-bundle lever.
- **Reuse.** Youtu's MLA is bit-identical to GLM-4.7-Flash's; the model overlay
  (`models/macos/youtu.py`) is `glm4_moe_lite` with the MoE FFN swapped for a dense `MLP` and a
  tied head, and the absorbed-MLA cache + flash-decode kernel (`glm4_moe_lite_absorbed.py`,
  `mla_metal_sdpa.py`) are reused verbatim (`kv_lora` 512 / `qk_rope` 64 / scale `192**-0.5` were
  already the DeepSeek-V2-Lite config the kernel bakes).
- **Thinking mode is verbose** — a simple question can spend many tokens reasoning before the
  answer; give it a generous token budget, or pre-fill an empty `<think>\n\n</think>` for
  No-Think replies.

## How to reproduce

```bash
cd coreai-models   # with the youtu model overlay (see ../conversion)
# convert (int8 body + int8 head, absorbed-MLA flash-decode kernel; ~2.2 GB bundle)
.venv/bin/python ../coreai-models-community/conversion/export_youtu_decode_pipelined.py \
    --hf-id tencent/Youtu-LLM-2B --split-g 8
# bench (pipelined engine)
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-runner \
    --model exports/youtu_llm_2b_decode_absorbed_msdpa \
    --raw-tokens rawtok.json --apply-chat-template false --sampling-strategy greedy \
    --max-tokens 16 --warmup exact --warmup-length 1
```

Model overlay: `models/macos/youtu.py` (naive dense-MLA + loader) + `youtu_absorbed.py` (absorbed
decode, reusing `glm4_moe_lite_absorbed`). **Decode-only loop-free** for the pipelined engine.
