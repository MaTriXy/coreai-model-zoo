# Qwen3.5-4B — Apple Core AI (`.aimodel`)

Qwen3.5-4B (the 4B member of the GDN hybrid linear-attention family) converted to Apple
**Core AI** for macOS 27 / iOS 27 (beta), riding Apple's **`coreai-pipelined` GPU engine**
via the same decode-only loop-free export as the
[0.8B](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) and
[2B](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) siblings — async encode,
on-GPU argmax sampling, on-device KV growth, zero custom kernels.

> [!NOTE]
> **b2-native repo (2026-07-15).** This bundle was exported with `coreai-core 1.0.0b2`
> and loads on the OS 27 **beta 3** toolchain. Unlike the sibling repos there is no
> June-era b1 tree here; `gpu-pipelined-b2/` is the only (and canonical) path.

## Bundles

- **`gpu-pipelined-b2/qwen3_5_4b_decode_int8hu_block32_sym/` — the ship config (~5.4 GB)**:
  transformer int8 linear per-block-32 + **untied 248K-vocab lm_head in per-block-32
  absmax int8** (`int8hu --head-sym`), the same head recipe validated on the 0.8B/2B ports
  (plain absmax `symmetric` — clipping variants flip oracle top-1s; full story in the zoo's
  [pipelined-engine notes](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/pipelined-engine.md)).
  Full LanguageBundle (`metadata.json` + `tokenizer/` + `.aimodel`), `input_ids` STATIC
  `[1,1]` single-step export → `EngineFactory` classifies it dynamic → pipelined engine.

## Measured — DeviceMark

Quality and speed for exactly these bytes are published on
**[DeviceMark](https://devicemark.github.io/)**: the full 596-item battery
(IFEval + MMLU + MATH) with Wilson CIs, retention vs the float baseline, and Mac decode
speed — see the qwen3.5-4B row, and per-entry gate provenance on the
[methodology page](https://devicemark.github.io/methodology.html).

⚠️ **Reasoning-style budgeting**: this model thinks at length before answering. Give it a
generous completion budget (DeviceMark evaluates it at **4096 max tokens**; tight caps get
eaten entirely by the thinking phase and yield empty answers).

## Run (macOS)

Needs the engine patch stack from the
[zoo](https://github.com/john-rocky/coreai-model-zoo) (`apps/coreai-shared-product.patch` →
`apps/coreai-pipelined-extra-states.patch`), then:

```bash
COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model qwen3_5_4b_decode_int8hu_block32_sym -p 128 -g 256 -n 3
```

- `COREAI_CHUNK_THRESHOLD=1` **before engine creation** — prefill runs as pipelined S=1
  steps (prompt tok/s ≈ decode tok/s).
- **Never call `engine.warmup()`** — it warms query length 256 and the static `[1,1]`
  graph rejects it. A 1-token generate after load is the warmup.
- Benchmark **Release** builds only (Debug measures ~3× slow).

## iPhone

No iPhone bundle is published here: 4B-class graphs exceed on-device GPU specialization
and need ahead-of-time (h18p) compilation. For phones, use the
[0.8B](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) (50+ tok/s in ~1 GB) or
[2B](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) (28–30 tok/s) pipelined bundles.

## Reproduce

Conversion script (self-contained) + method page in the zoo:
[`conversion/export_qwen3_5_decode_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_qwen3_5_decode_pipelined.py)
(`int8hu --head-sym --hf-id Qwen/Qwen3.5-4B`) ·
[`knowledge/pipelined-engine.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/pipelined-engine.md)

---

**⬇️ Download:** [🤗 mlboydaisuke/qwen3.5-4B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-4B-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show qwen3.5-4b` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
