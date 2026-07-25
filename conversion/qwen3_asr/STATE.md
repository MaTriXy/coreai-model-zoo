# Qwen3-ASR → Core AI — port state

**Target:** `Qwen/Qwen3-ASR-1.7B` (Apache-2.0) → Core AI `.aimodel`, iPhone-first, drop-in for
CoreAIKit. Zoo's **first ASR** (transcription) model — completes the speech stack
(Qwen-Omni understand + Kokoro TTS + **Qwen3-ASR transcribe**) on one runtime.

**Status:** **P0 + P1 + P2 + P3 DONE.** ✅ **CLEAN END-TO-END ON-ENGINE GATE PASSES** (`gate_static.py`):
encoder `.aimodel` → audio_embeds → static **prefill** `.aimodel` (seeds KV) → static **decode**
`.aimodel` greedy loop = **17/17 token-for-token vs golden** ("language Japanese<asr_text>音声認識の
テストです。今日はいい天気ですね。" + EOS). Decode is **FLAT ~41 ms/token** (~24 tok/s, compiles ONCE,
no wedge); prefill 0.6 s. The whole Core AI pipeline transcribes correctly on the GPU engine.
**P3.5 ✅ UNIFIED single bundle** (`export_unified.py`, `gate_static.py --unified`) — ONE 2.3 GB
`.aimodel` with `prefill`+`decode` entrypoints sharing ONE weight copy + one KV state.
**#2 ✅ VARIABLE-LENGTH (general transcriber, not ja1-only):** encoder re-exported at **K=30** (≤30 s,
engine cos 0.999997, host zero-pads mel→3000, trims 390→N) + **prefill is DYNAMIC in Sp/N** (one engine
specialization per length, **cached on disk**; ANE-probe-fail→GPU is a HARMLESS one-time ~1 s warmup
per new length, NOT the per-token wedge; decode stays fully static). CACHE_LEN=1024 (≤30 s audio +
transcript; cache only ~117 MB). Full pipeline gated **17/17 token-exact** (`gate_static.py --unified
--enc-chunks 30 --cache-len 1024`): K=30 enc → audio_embeds(55) → dynamic prefill 414 ms → static
decode flat ~57 ms/tok. **SHIP artifacts = `qwen3_asr_1.7b_audio_encoder_fp16_k30.aimodel` +
`qwen3_asr_1.7b_int8hu_unified_cl1024/`.** Next = **P4 iPhone** (sideload these two, measure RTFx).

## P3 artifacts (built, int8hu)
- `artifacts/qwen3_asr_1.7b_audio_encoder_fp16_k5.aimodel` (606 MB, engine cos 0.999995; K=5 ≤~10 s)
- `artifacts/qwen3_asr_1.7b_int8hu_unified_sp70_cl256/` (**2.3 GB, SHIP form** — prefill+decode fns)
- `artifacts/qwen3_asr_1.7b_int8hu_{prefill_sp70,decode_cl256}/` (2.3 GB each — two-bundle gate form)
- (removed the superseded `qwen3_asr_1.7b_decode_int8hu_n55{,_s1}` dynamic/wedged bundles to save
  disk; regenerate via `export_decoder.py` if ever needed — but the static bundles supersede them)

## P3 SOLVED — static prefill+decode + the externalized-RoPE trap (NEW files)
The wedge fix = **two fully-static graphs** (mirror `unlimited_ocr`), exported by `export_static.py`,
gated by `gate_static.py`, math in `qwen3_asr_static.py`:
  - **prefill** `Qwen3ASRStaticPrefill`: q_len=Sp (baked), V+slot audio injection, writes KV `[0,Sp)`,
    returns LAST-token logits (= decode step 0).
  - **decode** `Qwen3ASRStaticDecode`: q=1, `pos` a runtime int32 **VALUE** (not a shape) → write new
    K/V at slot `pos` via data-driven `mutable_slice_update`, read the WHOLE fixed buffer, explicit
    causal mask `j<=pos`. All shapes constant → engine compiles ONCE → flat decode.
  - Shared host KV state: the SAME `keyCache`/`valueCache` NDArray buffers are passed to the prefill
    fn then the decode loop (two separate bundles thread one cache, cf. unlimited_ocr gate).
**THE BUG that ate the day (isolation ladder injection→cache→small→1-attn; `debug_attn.py` kept, has
`--ext rope|rmsnorm|none` to re-verify the externalization decision for the 0.6B variant):**
the engine ran but transcribed garbage (first token 198="\n", logits cos 0.49, model ignoring audio).
Injection was perfect (cos 1.0), the KV-cache write/read-after-write/state-surfacing were perfect
(cos 1.0), 2-layer fp16 still failed → bisected to ONE op: **the engine-native EXTERNALIZED RoPE op
mishandles a BAKED-CONSTANT `position_ids` (`arange(Sp)`) in a fully-static graph** (layer-0 attn cos
0.69 externalized vs **1.0 decomposed**, error grows with position). FIX = **drop `rope` AND
`scaled_dot_product_attention` from `_EXTERNALIZE_SPECS`** for the static graphs (RoPE decomposes
bit-exact; SDPA must decompose anyway to take our explicit mask). RMSNorm/GatherMM stay externalized
(verified cos 1.0). NOTE the read-after-write hazard I first suspected was a RED HERRING — a tiny
float32 probe proved `mutable_slice_update`+immediate full-buffer read is exact on
the engine; the misleading "KVcache cos 0.0" was just fp16 `np.linalg.norm` OVERFLOW in the metric.

**Direction (decided):** Ship on Core AI / pipelined fast path, CoreAIKit drop-in. Do NOT compete
with FluidAudio on raw ANE speed (it already ships a legacy **Core ML + ANE** Qwen3-ASR for
iPhone/iPad). Differentiators: first on **Core AI** (new runtime), unified SDK alongside
chat/RAG/TTS/audio-understanding, 52 langs + dialects + music/song vs Apple SpeechTranscriber's
single-language on-device path. Variant order: **1.7B first**, then 0.6B (same config-driven script).

**Direction (decided):** Ship on Core AI / pipelined fast path, CoreAIKit drop-in. Do NOT compete
with FluidAudio on raw ANE speed (it already ships a legacy **Core ML + ANE** Qwen3-ASR for
iPhone/iPad). Differentiators: first on **Core AI** (new runtime), unified SDK alongside
chat/RAG/TTS/audio-understanding, 52 langs + dialects + music/song vs Apple SpeechTranscriber's
single-language on-device path. Variant order: **1.7B first**, then 0.6B (same config-driven script).

---

## Closest existing zoo precedents (fork these, don't start fresh)

This is essentially a **Qwen2.5-Omni thinker** with the audio tower swapped (AuT) and the decoder
swapped (Qwen3). Reuse, in priority order:

- `conversion/export_qwen2_5_omni_audio.py` + `models/.../qwen2_5_omni_audio.py` — static audio
  encoder export (Whisper mel recipe, fixed K chunks, host zero-pads mel). **AuT export mirrors this.**
- `conversion/unlimited_ocr/` — encoder `.aimodel` + decoder driven on `inputs_embeds` + masked_scatter,
  **NO engine patch** on the stock runtime. The glue pattern for Qwen3-ASR.
- `conversion/export_nanbeige41_decode_pipelined.py` / `export_qwen3_5_decode_pipelined.py` — plain
  Qwen3 dense decoder on the pipelined fast path. **Our decoder is exactly this.**
- `models/macos/qwen3.py` — reuse **verbatim** (see RoPE note below).

---

## Authoritative architecture (read from official `qwen_asr` source, NOT blog summaries)

Source of truth: `QwenLM/Qwen3-ASR` repo `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
(+ HF `Qwen/Qwen3-ASR-1.7B/config.json`). model_type `qwen3_asr`, wraps a `thinker_config`
(audio_config + text_config), like Qwen-Omni's thinker.

### Decoder — text_config, model_type `qwen3` → **standard Qwen3, reuse `qwen3.py` verbatim**
- 28 layers, hidden 2048, intermediate 6144, SwiGLU (SiLU), 16 Q / 8 KV heads, head_dim 128.
- **QK-RMSNorm** on head_dim (eps 1e-6), no attention bias, RMSNorm eps 1e-6, no biases.
- **tied** embeddings, vocab 151936, rope_theta 1e6.
- **RoPE: config says MRoPE (`interleaved:true`, `mrope_section:[24,20,20]`) BUT it is a NO-OP for
  ASR.** `get_rope_index` returns 3 **identical** position copies (plain `cumsum(attn_mask)-1` for
  prefill; `arange+delta` for decode). `apply_interleaved_mrope` with identical copies returns
  `freqs[0]` unchanged → cos/sin reduce **bit-exactly to standard 1D Qwen3 RoPE** with NeoX
  `rotate_half`. **So: feed sequential position_ids (0,1,2,…) and qwen3.py is correct as-is.**
  Do NOT author an interleaved-MRoPE variant; it is unnecessary.

### Audio encoder — audio_config, model_type `qwen3_asr_audio_encoder` (AuT), ~300M (1.7B variant)
- d_model 1024, 24 layers, 16 heads, FFN 4096, output_dim **2048** (= decoder hidden; projects
  straight into the embedding stream). num_mel_bins 128. `activation_function` = **gelu**.
  `scale_embedding` false. max_source_positions 1500.
- **Mel frontend:** WhisperFeatureExtractor — feature_size 128, n_fft 400, hop_length 160,
  sr 16000, chunk_length 30 (n_samples 480000, nb_max_frames 3000). **Same as Qwen2.5-Omni mel.**
- **Conv subsample (8×):** `conv2d1 Conv2d(1,480,k3,s2,p1)` → gelu → `conv2d2 Conv2d(480,480,k3,s2,p1)`
  → gelu → `conv2d3 Conv2d(480,480,k3,s2,p1)` → gelu. Then reshape `[b,c,f,t]→[b,t,c*f]` and
  `conv_out = Linear(480*16=7680, 1024, bias=False)`. (mel-bin 128 → //2 ×3 → 16.)
- **+ sinusoidal pos** (`SinusoidsPositionEmbedding(1500,1024)`, max_timescale 10000), added
  **per-chunk indexed from 0** (only first `tokens_per_chunk` positions per chunk).
- **Chunked windowed attention:** `n_window=50` → chunk = 100 mel frames → **13 tokens/chunk**
  (`_get_feat_extract_output_lengths`). Attention window = `13 * (n_window_infer 800 / 100) = 104`
  tokens (= 8 chunks); **bidirectional within each window block, masked across blocks** (built via
  `cu_seqlens`; is_causal=False). Encoder layer: pre-LN → MHA (bias on q/k/v/out) → residual →
  pre-LN → fc1→gelu→fc2 → residual. LayerNorm WITH bias.
- **Head:** `ln_post` (LayerNorm) → `proj1 Linear(1024,1024)` → gelu → `proj2 Linear(1024,2048)`.
- `conv_chunksize=500` is only an OOM split during conv, not architectural.

### Glue (Qwen3ASRThinkerForConditionalGeneration.forward)
1. `inputs_embeds = embed_tokens(input_ids)`.
2. `audio_features = audio_tower(mel)` → `[N_audio_tokens, 2048]` (run per-audio, no batch).
3. `mask = (input_ids == audio_token_id 151676)`; `inputs_embeds.masked_scatter(mask, audio_features)`.
4. sequential position_ids; prefill decoder on inputs_embeds; greedy decode to EOS.

### Tokens / template
- audio_start **151669**, audio_end **151670**, audio_pad/audio_token **151676**, `<asr_text>` **151704**.
- EOS **{151643, 151645}**, pad 151643. greedy (generation_config temp≈0, do_sample false).
- **Prompt template (CONFIRMED from `chat_template.json` + `processing_qwen3_asr.py`):**
  `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
  - `{system}` is **empty** by default. The template has ONE `<|audio_pad|>`; the processor
    expands it to **N** copies (N = audio token count from the encoder) per audio.
  - **language=None (auto):** assistant generates `{Language}<asr_text>{transcript}`; split via
    `parse_asr_output`. **Forced language:** prompt appends `language {Language}<asr_text>` to the
    assistant turn → text-only output. `context` is an optional biasing hint string.

---

## P0 results (done)
- Oracle harness: `conversion/qwen3_asr/make_oracle.py` → `oracle_golden.json`. Runs official
  `qwen_asr` (CPU/fp32) on `say`-generated 16 kHz clips.
- **Env (does NOT touch zoo exports):** the community venv
  `coreai-models-community/coreai-models/.venv` already has **transformers 4.57.6** (= qwen_asr's
  pin). Added only `librosa`, `soundfile`, `nagisa` (forced-aligner import dep). qwen_asr imported
  via sys.path from `/tmp/qwen3-asr-official` (not installed).
- Results: **ja1 PERFECT** ("音声認識のテストです。今日はいい天気ですね。"), lang-ID English/Japanese
  correct. en1/en2 partial — `say` synthetic speech is unclear; the model transcribed what it heard.
- **P1 parity target = the official model's GREEDY output tokens on the SAME audio** (deterministic),
  NOT the human-intended text. Enhance the oracle to also dump prompt ids + generated token ids
  (low-level `Qwen3ASRProcessor` + `Qwen3ASRForConditionalGeneration.generate`) for token-exact P1.

## P1 encoder results (done)
- `conversion/qwen3_asr/audio_encoder.py` — `AuTEncoderStatic` (export-friendly: mel `[1,128,100*K]`
  → `[K*13,2048]`, static `build_attn_bias(S,N,window=104)` replicates HF `cu_seqlens` windows +
  masks the partial last chunk's pad tokens, which always sit at flat `[N, K*13)`).
- `conversion/qwen3_asr/parity_encoder.py` vs `oracle_tokens.npz['encoder_out']`: **cos=1.00000203,
  max|Δ|=1.72e-6, mean rel 2.2e-6 → PASS** (ja1, K=5, N=55). Conv reshape, per-chunk sinusoid,
  windowed attention, head all bit-exact on first try.
- Confirms `_get_feat_extract_output_lengths`: 417 mel → 5 chunks → 4×13 + tail(17→3) = N=55.
- NEXT (decoder): map checkpoint `thinker.model.*` into `qwen3.py`, embed prompt ids, masked_scatter
  the 55 encoder tokens at audio_token positions, sequential position_ids, greedy decode → must match
  golden `gen_ids` `[11528,10769,151704,…,151645]` ("language Japanese<asr_text>…").

## P1 decode results (done)
- `conversion/qwen3_asr/parity_decode.py`: my static encoder → `inputs_embeds` masked_scatter →
  `thinker.generate(inputs_embeds=…)` greedy → **17/17 token match** vs golden `gen_ids`
  ("language Japanese<asr_text>音声認識のテストです。今日はいい天気ですね。" + EOS 151645).
- Confirms in practice: decoder = standard Qwen3 (qwen3.py math), sequential positions, audio
  tokens dropped in via masked_scatter at id 151676. No engine-specific behavior needed for parity.

## P2 progress
1. **Encoder → static fp16 `.aimodel` DONE.** `conversion/qwen3_asr/export_encoder.py` (export +
   self-gate via `export_to_coreai` + `rt.AIModel.load`). Inputs `input_features [1,128,100*K]` +
   `attn_bias [1,S,S]` → `audio_embeds [S,2048]`. **Engine gate (ja1, K=5): global cos 0.999995,
   per-token min 0.999893, max|Δ| 0.0011 → PASS.** Artifact `artifacts/qwen3_asr_1.7b_audio_encoder_fp16_k5.aimodel`.
   NOTE: eager-fp16 NaNs (the -65504 mask + fp16 matmul) but the exported graph/engine is correct —
   gate eager in fp32, engine in fp16. For ship bake K=30 (≤30 s clips; host zero-pads + trims).
   TODO: promote `AuTEncoderStatic` into `coreai_models/models/macos/qwen3_asr_audio.py`.
2. **Decoder** → map checkpoint `thinker.model.*`/`lm_head` into `qwen3.py`; export via
   `export_nanbeige41_decode_pipelined.py`-style pipelined int8 (head-sym), driven on `inputs_embeds`
   (mirror `unlimited_ocr/export_decoder.py`). Vocab 151936, tied embeddings, 28L/2048/16:8/hd128.
3. Host glue: mel (WhisperFeatureExtractor → reuse omni vDSP/np), prompt build, masked_scatter,
   greedy decode, parse `{lang}<asr_text>{text}`.
4. Gate: engine == torch token-for-token on ja1 (+ add en clips / a music clip).

## Plan & gates (zoo torch→python→engine→device method)

- **P0 oracle** — official `qwen_asr` (transformers 4.57.6) in a **dedicated venv** (do NOT pollute
  the zoo `.venv`). Make `say`-generated 16 kHz clips of known text (+ a multilingual + a music clip)
  → golden transcript + token-id sequence. Confirm exact prompt template + audio token count N.
- **P1 torch ladder** — eager re-impl: AuT encoder (zoo model style) + `qwen3.py` decoder + glue.
  Gate: torch eager == oracle token-for-token (greedy); mel cos≈1, encoder last_hidden cos≈1.
- **P2 Core AI export** — encoder → static fp16 `.aimodel` GraphModel (fixed K chunks, mel zero-pad;
  mirror omni audio). decoder → pipelined int8 (head-sym), inputs_embeds-driven (mirror unlimited_ocr).
  Gate: engine == torch token-for-token (fp16 ties via margin rule).
- **P3 host pipeline + Mac** — ✅ DONE. `gate_static.py` = static prefill+decode on engine, 17/17
  token-exact, decode flat ~41 ms/tok. The gate driver IS the host pipeline / CoreAIKit Transcriber logic.
- **P3.5 unify (for iPhone)** — ✅ DONE. `export_unified.py` = ONE 2.3 GB bundle, `prefill`+`decode`
  entrypoints sharing one weight copy + one KV state (`coreai_torch.TorchConverter` + two
  `add_pytorch_module`; quant in-place → shared quantized submodules). Gated 17/17 token-exact
  (`gate_static.py --unified`). Same RoPE/SDPA externalization drop as the two-bundle form.
- **P4 iPhone** — sideload (`devicectl copy --domain appDataContainer`); iPhone 17 Pro; ~2.5 GB peak
  (jetsam-safe), measure RTFx. AOT likely unnecessary at ~2B.
- **P5 CoreAIKit drop-in** — `Transcriber`/`KitASRModel` (mirror `KitAudioModel`); mic demo with
  **AVAudioRecorder** (NOT AVAudioEngine tap — known crash). offline ≤30 s first.
  **SHIP-FORM DECISION (locked 2026-06-23): OMNI-STYLE dynamic bundle + high-level engine, NOT the
  custom static bundle.** Why: the kit `AudioRuntime.swift:110` already drives a STANDARD dynamic
  Qwen decoder via `EngineFactory.createEngine(EngineOptions(staticInputBuffers: ["audio_embeds":
  StaticInputBuffer(buf)]))` + `engine.generate(promptIDs:)` — the SAME `audio_embeds` static buffer
  + V+slot trick, shipped on iPhone for Qwen-Omni. The high-level engine manages decode
  specialization correctly → **the wedge was ONLY my Python manual driver (growing position_ids), NOT
  a bundle defect.** So the ASR ship decoder = the DYNAMIC `export_decoder.py` bundle (regenerate;
  audio_embeds stays a graph input that the engine binds to the static buffer), driven by the
  high-level engine. The custom static prefill/decode bundle's job was DONE = it PROVED the engine
  numerics are token-exact (17/17), de-risking the dynamic ship. Kit `ASR/` mirrors `Audio/`
  (ASRRuntime+KitASRModel+KitASRExecutor+ASRPromptRenderer), reuses `AudioMelPreprocessor` (same
  Whisper mel) + the AuT K=30 encoder as a `GraphModel`. Gate on Mac via Swift (omni playbook).
  **P5 ✅ KIT MODULE WRITTEN (device-test pending, user):** `coreai-kit/Sources/CoreAIKit/ASR/`
  (5 files mirroring `Audio/`): `ASRArchitecture` (AuT K=30 geometry, per-chunk N, `[1,S,S]` bias
  port of `build_attn_bias`), `ASRRuntime` (loads `_s1` decoder + binds `audio_embeds` static buffer
  + AuT encoder GraphModel + `transcribe()` parsing `{lang}<asr_text>{text}`), `KitASRModel`
  (`ASRModelID.qwen3ASR1_7B`, direct `attach`/`transcribe` + `LanguageModel` conformance),
  `ASRPromptRenderer` (ASR template, V+slot rewrite), `KitASRExecutor` (session path, streams
  transcript). Reuses `AudioMelPreprocessor.qwen2_5Omni()` (same Whisper mel) — no Package.swift
  change (ASR/ is auto-included in the CoreAIKit target). SHIP decoder regenerated =
  `qwen3_asr_1.7b_decode_int8hu_n390_s1` (eager-int8 first-token=golden).
  **P6 prep ✅:** zoo card `zoo/qwen3-asr.md` written.
  **APP ✅ WIRED (device-test pending):** zoo `apps/coreai-audio` got a **Transcribe** tab
  (`TranscribeModel.swift` + `TranscribeView.swift`, added to the TabView). `project.yml` points
  coreai-kit at the LOCAL checkout (`path:` = your `coreai-kit` clone, to pick up the
  uncommitted `ASR/` module — revert to URL+revision after committing the kit) and was regenerated
  with `xcodegen`. macOS loads the decoder/encoder straight from the local conversion artifacts
  (`TranscribeModel.macArtifacts`); iOS uses the sideload-to-skip-DL pattern. TEST: open
  `coreai-audio.xcodeproj` → build (Release) → Transcribe tab → Load model → Record/Choose →
  Transcribe.
  **✅✅ MAC END-TO-END VERIFIED 2026-06-23:** the built app's ASR self-test (`ASRSelfTest.swift`,
  `ASR_SELFTEST=1` in `App.init`) transcribed the `say`-generated ja1 clip EXACTLY: `lang=[Japanese]`
  + `音声認識のテストです。今日はいい天気ですね。` in **1994 ms** (warm). Proves the whole assembled Swift
  path (KitASRModel → AuT encoder → audio static buffer → high-level engine driving the `_s1` decoder
  → `{lang}<asr_text>{text}` parse) AND that the **omni-style ship form does NOT wedge** (architecture
  decision confirmed). BUILD SUCCEEDED (one fix: `convertTokenToId` already returns Int? → drop
  `.map(Int.init)`). Headless-run gotchas (Mac CLI, NOT the user's Xcode flow): GUI app needs a
  WindowServer session → launch via `open`; MisakiSwift.framework rpath is `@executable_path/Frameworks`
  but it ships in `Contents/Frameworks` + SIP strips `DYLD_*` from launchd → copy it to
  `Contents/MacOS/Frameworks` for a standalone launch (Xcode Cmd+R sets the search paths, user flow OK).
  Remaining = iPhone device test (user) + commit kit ASR + HF upload (confirm).
- **P6 ship** — HF `mlboydaisuke/Qwen3-ASR-1.7B-CoreAI` (encoder/ + decoder/ + assets/ + tokenizer/,
  Apache-2.0) + zoo card `zoo/qwen3-asr.md` + conversion script + README row + knowledge note.
  0.6B falls out of the same config-driven script (encoder d896/18L/14h, FFN 3584, output_dim 1024).

## Risks / open items
- **Chunk/window attention** is the fiddly part — replicate `cu_seqlens` block mask exactly; for a
  static graph bake K chunks (windows of 8 chunks / 104 tokens). Cross-check vs P0 encoder output.
- **AuT is encoder-only at inference** (its AED-pretrained decoder is dropped; Qwen3 is the decoder).
  Confirmed: model uses `audio_tower` (encoder) → masked_scatter → Qwen3 decoder.
- offline ≤30 s first; streaming (dynamic 1–8 s window) + timestamps (`Qwen3-ForcedAligner-0.6B`) = v2.
- Confirm `inputs_embeds` runs on stock pipelined runtime w/o patch (unlimited_ocr proves it; else
  fall back to the static-input hook already in the repo). [P3 confirms: in-graph V+slot injection
  works on the engine, cos 1.0 — no patch needed; audio rides a regular `audio_embeds` input.]
- **Variable-length prefill (ship):** the static prefill is baked at `Sp` (here 70 = 55 audio + 15
  template). Each audio length → different N → different Sp → a different prefill graph. For ship,
  either (a) bake a fixed MAX `Sp` for ≤30 s audio (max N≈390 → Sp≈405) + pad the prompt with pad
  tokens + extend the causal mask to also mask pad columns (one graph for all clips, mirrors the
  K=30 encoder pad/trim), or (b) keep per-length prefill specialization (one-time recompile per new
  length, cached). The **decode** graph is fully length-agnostic (works as-is). Decide at P3.5/P4.

## Paths
- weights: HF cache `~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B` (downloaded).
- reference impls (ephemeral): `/tmp/qwen3-asr-official` (official), `/tmp/qwen-asr-antirez` (annotated C).
- config snapshot: `/tmp/qwen3asr_1.7b_meta/config.json`.
- port dir: `conversion/qwen3_asr/` (this file).
