# Running Gemma 4 (with its PLE table) behind FoundationModels

Gemma 4's small models (E2B / E4B) carry a non-standard extra input: a giant **per-layer
embedding (PLE) table** the decoder gathers from once per token — the text-model analogue of a
VL decoder's `image_embeds`. A stock text load path declares only `input_ids` / `position_ids`
(+ KV states), so it can't feed the PLE table and the engine rejects the bundle. This page shows
how CoreAIKit runs Gemma 4 behind Apple's FoundationModels `LanguageModel` protocol — "Ask
Gemma 4" through a plain `LanguageModelSession` — by binding the PLE table as a **constant static
graph input**. (Sibling: [vl-executor-fm-stack] for the VL version of the same trick;
[pipelined-engine] for the engine APIs.)

## The bundle shape that makes it easy: `…_decode_int4lin_tbl`

The published pipelined Gemma 4 bundle comes in two forms (see
[pipelined-engine] "per-token vs static inputs"):

- **provider mode** — a per-token `PerTokenInputProvider` fills the PLE gather each step.
- **tbl / static mode** — the PLE table is exported as a **graph input** (`ple_table`
  [vocab, ld·layers] int8 + `ple_scale` [vocab] f32), gathered **in-graph** by `index_select` on
  `input_ids` (the `q·s·√ld` scaling is in-graph, bit-exact). The head (tied lm_head + final
  softcap) is fused into the same graph, so the engine samples logits directly.

**Use the `…_tbl` bundle.** Then the host only binds two buffers that never change for the life of
the model — no per-step host work, no encoder, no per-turn rewrite. It is the simplest possible
wiring: a plain text bundle plus two constant inputs.

## The wiring (≈ the whole runtime)

```swift
// 1) S=1 prefill: set BEFORE the engine reads ModelConfig.chunkThreshold.
setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

// 2) Read each PLE table file once into an OWNED storageModeShared MTLBuffer (owned beats a
//    read-only mmap here — a no-copy mapping pays a large per-encode residency tax, and these
//    are bound on every step). ~2.35 GB for E2B.
let buffers: [String: StaticInputBuffer] = [
  "ple_table": StaticInputBuffer(ownedBuffer(tablesDir + "/embed_per_layer.i8")),
  "ple_scale": StaticInputBuffer(ownedBuffer(tablesDir + "/embed_per_layer.scale.f32")),
]

// 3) Build the engine through the factory so the static inputs can be bound (the default
//    CoreAIRunner load path does not expose EngineOptions). "main" avoids CoreAIShared's
//    ComponentKey.
let bundle = try LanguageBundle(at: decoderDir)
let config = ModelConfig(name: bundle.name, tokenizer: bundle.tokenizer,
  vocabSize: bundle.vocabSize, maxContextLength: bundle.maxContextLength,
  serializedModel: [bundle.modelAssetPath], function: "main")
let engine = try await EngineFactory.createEngine(
  config: try JSONEncoder().encode(config),
  modelURL: try bundle.requireModelURL(for: "main"),
  options: EngineOptions(staticInputBuffers: buffers))   // ← the one line that unblocks Gemma
```

After this, generation is identical to any pipelined text bundle: `engine.reset()` →
`engine.generate(with: promptTokens, …)` → stream tokens, on-GPU argmax, on-device KV. Keep the
two buffers alive for the engine's lifetime. Do **not** call `engine.warmup()` (it warms query
length 256, which the S=1 graph rejects — a 1-token generate after load is the warmup). A QAT
bundle must be paired with the **QAT** PLE tables.

## Gemma 4's chat format (it is NOT Gemma 3's)

Gemma 4 has a new tokenizer. Turns are framed by `<|turn>` (105) … `<turn|>` (106) — **not**
`<start_of_turn>`/`<end_of_turn>`. Reasoning rides a `<|channel>thought\n … <channel|>` (100 …
101) channel; `<eos>` = 1, `<bos>` = 2. The shipped E2B bundle uses the stock
`google/gemma-4-E2B-it` tokenizer, which has **no embedded chat template**, so render the prompt
explicitly:

```
<bos>
<|turn>system\n{system}<turn|>\n        # only if instructions are present
<|turn>user\n{user}<turn|>\n
<|turn>model\n                          # generation prompt
```

Two gotchas, both measured on `gemma4_e2b_qat_decode_int4lin_tbl`:

- **Emit `<bos>` yourself** — Gemma's tokenizer post-processor does not add one.
- **Do NOT pre-inject an empty `<|channel>thought\n<channel|>` "thinking-off" channel.** The
  jinja's thinking-off path injects it, but on this bundle that *triggers* a verbose reasoning
  block; ending the prompt at plain `<|turn>model\n` yields a direct answer. Let the model open
  its own reasoning channel when it wants to, and route it with an OutputProfile (thinking rule
  `<|channel>thought\n` … `<channel|>`, `.stream` default) so a direct answer streams as response
  and a reasoning span streams as `.reasoning`. Stop generation on `<turn|>` (106) and `<eos>` (1).

## Memory / device

- **E2B `…_tbl` ships Mac + iPhone.** iPhone 17 Pro: 2.35 GB tables → ~4.4 GB peak vs the ~6.44 GB
  entitled jetsam limit (needs the increased-memory entitlement). The ~2 GB-constants graph
  crashes the on-device specializer → ship the **AOT `…_tbl_aotc_h18p` `.aimodelc`** on device
  (see [aot-and-specialization]). Verified: greedy 8/8 vs HF, decode 30.3 tok/s / prefill 38.9
  (settled), token-identical to the M4 Max GPU.
- **E4B `…_tbl` is Mac-only** (Mac decode 55.8 tok/s, 8/8). On iPhone the static-table form
  exceeds the budget; the device E4B path is **provider mode** (mmap PLE, ~2.2 GB footprint,
  decode ~15 tok/s) — the first 4B-class Gemma on an iPhone in this project.

## Net

The only thing standing between Gemma 4 and the standard text path is two extra graph inputs. Bind
them as constant static buffers and Gemma 4 runs behind `LanguageModelSession` like any other
local model — so "Ask Gemma 4" from Siri / App Intents is just a model swap. Same `EngineOptions`
hook the VL executor uses for `image_embeds`; here it carries a per-layer-embedding table instead.
