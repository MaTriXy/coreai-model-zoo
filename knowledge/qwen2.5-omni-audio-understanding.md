# Qwen2.5-Omni-3B as on-device audio *understanding* (a dynamic encoder made static)

Qwen2.5-Omni's **Thinker** (perceive + reason → text) ported to Core AI as the zoo's first on-device
**audio understanding** model — it describes the *sounds* it hears (events, texture, speech as "a man
speaking in English"), it is **not** a transcriber. The speech-out **Talker** is not ported. It runs
as two `.aimodel` bundles on the `coreai-pipelined` GPU engine (sibling: [pipelined-engine]). The
hard parts were all about making a model whose every interesting piece is *dynamic* fit a fixed-shape
engine — and the payoff (the AOT memory win + the ANE-reachable encoder) is below.

Backbone facts: decoder = vanilla **dense Qwen2.5-3B** (36L, hidden 2048, 16q/2kv, head_dim 128,
QKV-bias, SwiGLU, RoPE θ1e6, **untied** lm_head, vocab 151936), int8lin. Audio tower =
Whisper-large-v3 style (128-mel, d_model 1280, 32L, 20h), with the **projection built into the tower**
so `last_hidden_state` is already `[N, 2048]`. Checkpoint prefixes: `thinker.model.*` +
`thinker.lm_head.weight` + `thinker.audio_tower.*` (talker/token2wav skipped).

## The split: encoder bundle + decoder bundle, audio injected as a buffer

| bundle | what | size |
|---|---|---|
| `…_audio_encoder_fp16_k15.aimodel` | Whisper-style encoder, K=15 chunks ≈ 30 s — both platforms | 1.2 GB |
| `…_thinker_int8lin_n750_s1.aimodel` | text decoder, S=1 static query, `audio_embeds [750,2048]` buffer — macOS | 3.9 GB |
| `…_thinker_n750_ios/…aimodelc` | the decoder **AOT** (h18p) for iPhone | 4.5 GB |

The decoder takes the audio embeds on **one static-input buffer** (`audio_embeds [750,2048]`); the
prompt's `<|AUDIO|>` placeholders carry extension ids `V + slot` that the graph gathers in-graph
(`where(ids >= V, audio_embeds[ids - V], embed(ids))`) — the same VL trick as the image executor
(siblings: [fm-provider], the VL executor stack). FoundationModels has **no audio attachment** (its
swiftinterface only declares image), so audio is attached **out of band**: the app calls
`model.attach(samples:)` (mel → encoder → buffer) *before* `session.respond`, and the executor
rewrites N×`<|AUDIO|>` → `V…V+N-1`. Prompt framing: `<|audio_bos|>` + N×`<|AUDIO|>` + `<|audio_eos|>`
+ question.

**`N` and `K` do not grow the bundles.** The encoder is 1.2 GB at K=2 and at K=15 (the 32 layers'
weights are shared across chunks; only activations `[K,100,1280]` scale). The decoder is 3.9 GB at
N=100 and at N=750 (`audio_embeds` is an *input*, not weights; padded rows `[N_real,750)` are masked
out and decode bit-identically). So size the bundle for the max once; padding is free at the weight
level.

## The headline rework: ragged chunk attention → batched fixed attention

HF's `Qwen2_5OmniAudioEncoder.forward` is dynamic: the mel is split into `n_window*2 = 200`-frame
chunks; per chunk, conv1 + conv2(stride2) → 100 frames; attention is **block-diagonal = within-chunk
only** via `cu_seqlens`; then a whole-audio `avg_pool`(stride2) + `ln_post` + `proj` → 2048. None of
`cu_seqlens` / data-dependent `.split` / boolean-mask indexing exports to a fixed graph.

The realization that unlocks it: **that within-chunk block-diagonal attention *is* just a batched
full attention over `[K, 100, 1280]`** — one chunk per batch row = one fixed graph. Every other op
(conv, LN, q/k/v/o/fc linears) is per-frame and layout-invariant, so the static graph is
**numerically identical** to the ragged original (torch static-vs-eager cos **1.000000**, full *and*
partial-chunk).

**Only one mask is needed** — `attn_bias [K,1,1,100]` on the post-CNN axis for the partial last
chunk. There is provably **no conv mask** (conv2's last valid frame at col `v-1` reads mel cols
`[2v-3, 2v-2, 2v-1]`, all in range, so a trailing zero-pad never reaches a valid frame) and **no
avg_pool boundary leak** (the one contaminated pooled token lands at output index ≥ N and the host
trims it). Fully-padded trailing chunks are fully masked so softmax stays finite. Host contract:
zero-pad mel → `[1,128,3000]`, build `attn_bias`, run, trim `[N,2048]` by `feature_attention_mask`,
zero-pad to `[750,2048]`. Gates: fp16-torch 0.999964; engine GPU 0.999985/0.999979/0.999901
(noise/tone/beeps).

> Converter trap: `AvgPool1d` on an unbatched `[C,L]` makes `replace_avg_pool2d` throw
> `IndexError: shape[3]`. Fix = `reshape(N,2,d).mean(1)` (pool axis is even → bit-identical).

## TMRoPE collapses to 1-D for audio+text

Omni uses TMRoPE (time-aligned multimodal RoPE) for *video*. With audio+text only, the oracle's
`get_rope_index` skips the vision branch → `t == h == w` sequential → **standard 1-D RoPE is
bit-exact** (verified `rope_coords_equal == True`). So `from_hf` nulls `rope_scaling` and the engine
uses its native `arange` — none of the VL port's 4 inputs + 2 rope-shift scalars. A scary multimodal
positional encoding that degenerates to vanilla in your sub-case; check before you implement it.

## The mel front end in Swift, bit-exact

The 16 kHz log-mel is Whisper-large-v3 (`feature_size 128, n_fft 400, hop 160`). `n_fft = 400` is
**not a power of two**, so a radix-2 vDSP FFT can't do it → the per-frame DFT is a precomputed
**cos/sin matmul** (`vDSP_mmul`) — exact. Recipe: reflect-pad `n_fft/2 = 200`, Hann *periodic*,
`torch.stft` center then drop the last frame, `log10(max(., 1e-10))`, `max(., globalMax - 8)`,
`(. + 4)/4`. The `mel_filters` matrix is `[201,128]`, dumped once from HF and **bundled as a
CoreAIKit resource** so it never re-derives librosa. Gate vs the HF extractor: cos **1.0**.

## Memory / device — the AOT clean-mmap win

The surprise: a 3.9–4.5 GB decoder is **comfortable** on a 12 GB iPhone, not tight. After loading
both models on an iPhone 17 Pro: **5930 MB still free**. Reason: **AOT `.aimodelc` weights mmap as
clean, file-backed pages → they do not count against the jetsam *dirty*-memory limit** (only the
encoder JIT + activations + KV are dirty, ~1.5 GB). So "AOT vs JIT" is not just compile-time/spike
(see [aot-and-specialization]) — it decides *which* memory limit you hit. Consequence: **no
phase-split, no int4 needed**; the GPU path has headroom and so will the ANE path.

iPhone recipe: decoder = AOT (`coreai-build compile --platform iOS --architecture h18p
--preferred-compute gpu --expect-frequent-reshapes` → 4.5 GB; dodges the 3.9 GB on-device JIT
jetsam); encoder = JIT 1.2 GB (smaller than its 2.4 GB AOT, JITs fine); needs the
`com.apple.developer.kernel.increased-memory-limit` entitlement; delivery = sideload to
`Documents/models/` via `devicectl … copy --domain-type appDataContainer`. A19 Pro: white-noise →
"I hear a loud hissing sound." (drops "continuous" vs Mac = a benign AOT/fp16 greedy synonym fork);
mic → "I hear a man speaking in English." Mac M4 Max on the Swift engine: load 1.0 s, encode
0.16–0.18 s, gen ~1 s. (Companion TTS port: [kokoro-tts], same `coreai-audio` app, "Speak" tab.)

> Shipping trap: a zoo app that bundles **raw SPM resource bundles** (swift-transformers etc.) +
> a custom entitlement will **not device-codesign** (Xcode embeds them after the run-script phases;
> `--deep` over-applies the entitlement). Fix = depend on the **CoreAIKit library product**
> (`KitAudioModel`), which seals those bundles as *resources* that sign cleanly.

## ANE — reachable via the encoder, and 0.99 cos is enough

The int8 **dynamic** decoder graph can't compile on the ANE at all (`MLIR MPS to ANEC conversion
failed`). But the **static-shape encoder compiles and runs on the Neural Engine** at cos ~0.99
(0.9897/0.9971/0.9968 = 32-layer fp16-accumulation drift) — and at that 0.99 it decodes to
**byte-identical text** for all three contents. So **0.99 cos was good enough downstream; no
precision mitigation (palettize / high-precision softmax) was needed** — verify the end-to-end
*text*, not just the cos. Fixed shape is the prerequisite for ANE; the remaining lever is the
decoder's own static-shape ANE rework (the energy/mJ-per-token gate is the thesis proof).

## Runtime traps (the python loop — all gone on the real engine)

These hit when driving the decoder per-token from the **python runtime** (S=1, position_ids grows →
a new MPSGraph shape every step). On the actual pipelined engine (dynamic-shape native) and under the
static-shape rework, they vanish:

- The python runtime **cannot run a dynamic-output graph** → export the `_s1` (static `[1,1]` query,
  dynamic position_ids/KV) twin for gating. The `_s1` twin is also **what ships on the Swift engine**
  with `COREAI_CHUNK_THRESHOLD=1`; the fully-dynamic twin crashes the macOS-beta engine
  ("NSArrayM nil insert"). Feed tokens **S=1, not full-prefill** (full prefill → MPSGraph dtype assert).
- Metal corruption at two timescales: within a rollout ~step 25 (`MTL4CommandQueueError`) → **reload
  the bundle every 8 calls** (python-held KV NDArrays survive the reload); cumulative across ~400+
  calls even *with* reloads → decode **one clip per fresh process**.
- `~/Library/Caches/coreai-cache` caches a multi-GB compiled graph **per shape** → it hit **423 GB**,
  filled the volume, SIGSEGV "No space left on device". Purge before long dynamic-shape runs.
- Export is **CPU + disk only** (`to_coreai` / `optimize` / `save_asset` never touch GPU/ANE) → a
  re-export can run *concurrently* with a GPU decode job as long as it writes a different filename.

## Net

Qwen2.5-Omni's Thinker is a vanilla dense decoder + a Whisper-style encoder, so the work was almost
entirely *de-dynamizing*: rewrite the encoder's ragged chunk attention as one batched fixed attention
(bit-exact, one mask), null TMRoPE down to 1-D, match the mel in vDSP as a DFT matmul, and feed audio
embeds on a static buffer with extension-token gather. The two findings worth carrying forward: AOT
clean-mmap weights dodge the iOS jetsam dirty limit (so big bundles fit), and a fixed-shape encoder
runs on the ANE where the dynamic decoder can't — at a cos that *looks* lossy but is byte-identical
downstream.
