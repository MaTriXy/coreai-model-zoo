---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-2.6B
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - lfm2
  - hybrid
pipeline_tag: text-generation
---

# LFM2.5-2.6B — Apple Core AI (`.aimodel`)

**LiquidAI's LFM2.5-2.6B converted to Apple's Core AI** (the Core ML successor announced at
WWDC26), ready to run on iOS 27 / macOS 27. Same conv + full-attention hybrid as the
[1.2B](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI), scaled to **30 layers = 22
short-conv mixers + 8 GQA attention layers**, hidden 2048, MLP 10 752, 32 q / 8 kv heads,
vocab 128 000. No recurrent scan anywhere, so the decode graph is loop-free by construction and
rides Apple's **`coreai-pipelined` GPU engine** with one fixed-shape conv state and no custom
kernels.

This is a **reasoning model** — the chat template ends the generation prompt with an open
`<think>`, and generations spend their first few hundred tokens thinking. Budget `max-tokens`
accordingly; a 200-token cap regularly ends mid-thought.

> Requires the iOS 27 / macOS 27 beta (Core AI ships with the OS). Conversion code, gates and
> knowledge base: **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | prompt tok/s | decode tok/s | oracle gate |
|---|---:|---:|---:|---|
| `gpu-pipelined/lfm2_5_2_6b_decode_int8hu_block32_sym` | 3.4 GB | 139.5 | **116.7** | **PASS 16/16** |
| `gpu-pipelined/lfm2_5_2_6b_decode_int4lin` | **2.0 GB** | 170.6 | **139.2** | **PASS 16/16** |

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`. The gate compares the exported
bundle's greedy decode token-for-token against the fp32 eager oracle; transcripts are in the
[zoo card directory](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/lfm2.5-2.6b).

**No iPhone numbers are published here because none were measured.**

`int8hu` is the quality ship. The head is 128 000 × 2048 = 262 M parameters, so leaving it fp16
costs 524 MB of reads per token; untying and quantizing it to int8 buys **+8.6 % decode** over
plain `int8lin` (the 1.2B saw +9.3 % for the same reason). The bundle gets *bigger* — 3.2 → 3.4 GB
— because untying stores an fp16 embedding and an int8 head instead of one shared table. That is
the trade working, not a regression. `int8lin` is not published: slower than `int8hu` and only
0.2 GB smaller, so it has no case of its own.

`int4lin` did not hit the int4 quality cliff, which is worth saying because this family usually
does. Beyond the 16/16 gate, four long greedy generations were read in full — an algorithmic
explanation, iterative Fibonacci with complexity analysis, a clock-arithmetic word problem and a
Japanese instruction — with grammar intact, arithmetic correct and code correct. Four prompts are
not a benchmark: read this as *no cliff observed*, not *int4 is free*.

If you are sizing for a phone, note that 524 MB of the 2.0 GB `int4lin` bundle is the still-fp16
embedding. The remaining size lever on this model is the embedding, not the layers.

## Run it

```bash
git clone https://github.com/apple/coreai-models   # + the zoo's engine patches, see below
swift build -c release --product llm-runner

COREAI_CHUNK_THRESHOLD=1 .build/release/llm-runner \
  --model gpu-pipelined/lfm2_5_2_6b_decode_int8hu_block32_sym \
  --prompt "Explain why a hash table lookup is O(1) on average but O(n) in the worst case." \
  --max-tokens 512 --sampling-strategy greedy \
  --inference-engine-variant coreai-pipelined --warmup off
```

`--warmup off` matters: default warmup submits a synthetic 256-token prefill, and these bundles
are static-S=1, so it fails with a shape-substitution error before generating anything. The
`coreai-pipelined-extra-states` patch (which carries the conv state) is in the zoo under `apps/`.

## Converting this family yourself

Two config traps in this checkpoint generation, both silent — nothing raises, and a bundle built
without the fix looks like it worked:

1. **RoPE theta moved.** These are transformers-v5-era configs carrying
   `rope_parameters: {rope_theta: 1e7}` instead of a flat `rope_theta`. Read only the flat key and
   you fall back to the 1.2B's `1e6` and mis-rotate every position, with no error.
2. **The tokenizer class does not exist yet.** `tokenizer_config.json` declares
   `tokenizer_class: "TokenizersBackend"`, which a transformers-4.x `AutoTokenizer` cannot resolve.
   The chat template also lives in its own `chat_template.jinja` in this era, so the obvious
   workaround ships a bundle with no template. Load `tokenizer.json` directly and carry the
   template across.

Both are handled in
[`conversion/export_lfm2_decode_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_lfm2_decode_pipelined.py).

## License

LFM Open License v1.0, carried from
[`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B) (revision
`ab00687315bc1298e9d54e9c4b611dde9867ccc2`). Not affiliated with Apple or LiquidAI.
