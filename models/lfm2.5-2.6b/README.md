# LFM2.5-2.6B (text decoder) — Core AI

[`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B) is the large sibling of the
[LFM2.5-1.2B](../lfm2.5/README.md) this repo already ships: the same conv + full-attention hybrid,
scaled to **30 layers = 22 short-conv mixers + 8 GQA attention layers**, hidden 2048, MLP 10 752
(SwiGLU, *not* auto-adjusted — see below), 32 q / 8 kv heads, head_dim 64, vocab 128 000, tied head.
Like the 1.2B it has no recurrent scan anywhere, so the decode graph is loop-free by construction and
rides the existing [pipelined-engine fast path](../../knowledge/pipelined-engine.md) with one
fixed-shape conv state `[22, 1, 2048, 2]`.

Unlike the 1.2B it is a **reasoning model**: the vendor chat template ends the generation prompt with
an open `<think>`, and generations spend their first few hundred tokens thinking. Budget
`max-tokens` accordingly — a 200-token cap frequently ends mid-thought.

Source checkpoint: revision `ab00687315bc1298e9d54e9c4b611dde9867ccc2`, config SHA-256
`480f63fa8e1efa534ae8b92774b3b53b8d6812d62a726e9ecfc866933662f273` (LFM Open License v1.0).

**⬇️ Converted `.aimodel` bundles (ready to run):
[mlboydaisuke/LFM2.5-2.6B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-2.6B-CoreAI)** —
`gpu-pipelined/lfm2_5_2_6b_decode_int8hu_block32_sym/` (quality ship) and
`gpu-pipelined/lfm2_5_2_6b_decode_int4lin/` (2.0 GB), both full LanguageBundles including the
tokenizer and chat template, shipping the upstream LFM Open License v1.0.

## Verified

Mac numbers below were measured on this machine; **no iPhone numbers are published because none were
measured** — the device leg is open as a [device gate request](../../../../issues/new?template=device-gate-request.yml).

| bundle | size | prompt tok/s | decode tok/s | oracle gate |
|---|---:|---:|---:|---|
| `int8lin` | 3.2 GB | 125.4 | 107.4 | **PASS 16/16** |
| `int8hu --head-sym` | 3.4 GB | 139.5 | **116.7** | **PASS 16/16** |
| `int4lin` | **2.0 GB** | 170.6 | **139.2** | **PASS 16/16** |

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1` / `coremltools 9.0`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`. Gate =
`conversion/coreai_gate.py`, greedy, warmup off, engine variant `coreai-pipelined`, compared
token-for-token against the fp32 eager oracle; transcripts are in this directory.

**`int8hu` is the quality ship** and repeats the 1.2B's result for the same reason: the head is
128 000 × 2048 = 262 M parameters, so leaving it fp16 costs 524 MB of reads per token. Untying it and
quantizing to int8 halves that and buys **+8.6 % decode** (the 1.2B saw +9.3 %). The bundle gets
*bigger* — 3.2 → 3.4 GB — because untying means storing an fp16 embedding and an int8 head instead of
one shared table. That is the trade working as intended, not a regression.

**`int4lin` did not hit the int4 cliff**, which is worth stating because this family usually does.
Beyond the 16/16 gate, four long greedy generations (algorithmic explanation, iterative Fibonacci with
complexity analysis, a clock-arithmetic word problem, and a Japanese instruction) were read in full:
grammar intact, arithmetic correct, code correct. Compare `lfm2.5-8b-a1b-moe`, gemma4-12B and the
Qwen int4 attempts, which flip a high-margin token and degrade into broken repetition. Four prompts
are not a benchmark — treat this as "no cliff observed", not "int4 is free".

For iPhone the interesting number is the 2.0 GB `int4lin`, but note the fp16 embedding is 524 MB of
it: the remaining size lever on this model is the embedding, not the layers.

Port write-up, including what generalizes to the rest of this checkpoint generation:
[`knowledge/lfm2.5-2.6b-port.md`](../../knowledge/lfm2.5-2.6b-port.md).

## Two config traps this port had to fix

Both are silent — nothing raises, and a bundle built without the fix looks like it worked.

1. **RoPE theta moved.** This checkpoint is transformers-v5 era and carries
   `rope_parameters: {rope_theta: 1e7}` instead of a flat `rope_theta`. `lfm2_config_from_dict`
   read only the flat key and would have fallen back to the 1.2B's default `1e6` — every position
   mis-rotated, no error. The parser now reads both, and `tie_word_embeddings` alongside
   `tie_embedding` for the same reason.
2. **The tokenizer class does not exist yet.** `tokenizer_config.json` declares
   `tokenizer_class: "TokenizersBackend"`, which a transformers-4.x `AutoTokenizer` refuses to
   resolve — *after* the `.aimodel` has been written, leaving a bundle that is one directory short of
   usable. The chat template also lives in its own `chat_template.jinja` in this era, so a naive fix
   ships a bundle with no template at all (`zoo_verify.py` fails a published repo for exactly that).
   `export_lfm2_decode_pipelined.py:save_tokenizer` loads `tokenizer.json` directly and carries the
   template across. The saved tokenizer was gated against the source `tokenizers` library on six
   round-trip cases including Japanese and emoji.

## Reproduce

```bash
python3 conversion/zoo_convert.py run lfm2.5-2.6b
# or directly:
coreai-models/.venv/bin/python conversion/export_lfm2_decode_pipelined.py \
    int8hu --head-sym --hf-id LiquidAI/LFM2.5-2.6B
python3 conversion/coreai_gate.py exports/lfm2_5_2_6b_decode_int8hu_block32_sym \
    LiquidAI/LFM2.5-2.6B -n 16
```

The export needs the `lfm2` model overlay on the `coreai-models` checkout and the pipelined-engine
extra-states patch to *run* the bundle; `conversion/zoo_convert.py doctor` checks both.
