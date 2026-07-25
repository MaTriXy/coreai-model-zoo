# LFM2.5-Audio — unified speech↔speech (ASR + TTS) on Core AI (port notes)

Verified engineering notes from porting [`LiquidAI/LFM2.5-Audio-1.5B`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
to Core AI — the zoo's **first unified speech↔speech audio-language model** (one model understands *and*
speaks; no separate ASR/TTS). Moshi-class: a FastConformer audio encoder + the shipped LFM2-1.2B backbone +
a small RQ "depthformer" audio head + a custom LFM detokenizer. Both directions verified on the Mac Core AI
engine, byte-identical to the official `liquid_audio` reference.

License: **LFM Open License v1.0** (same as the already-shipped LFM2.5-1.2B → redistribution precedented).
The Mimi tokenizer weights are CC-BY-4.0 (only needed to *encode* reference audio, not for TTS).

## The model is one backbone + two heads + host DSP

| component | role | Core AI form |
|---|---|---|
| **preprocessor** | wav 16 kHz → 128-mel log-mel (NeMo filterbank) | host DSP (Nemotron/Sortformer recipe) |
| **audio_encoder** | FastConformer 17L d512 (dw-striding ×8 → 12.5 Hz, rel-pos MHA) | static graph, GPU cos 0.999999 |
| **audio_adapter** | LN→Linear→GELU→Linear (512→2048) | folded into the encoder graph |
| **lfm** | LFM2-1.2B backbone (2048d, 16L = 6 attn + 10 conv), **embeds-input** | stateful S=1 decode graph (AOT `.aimodelc`) |
| **depth_linear** | 2048 → 8×1024 (per-codebook depthformer inputs) | host (tiny matmul) |
| **depthformer** | 6L transformer, GQA 32Q/8KV hd32, qk-norm, 8-step RQ AR loop | **host** (eager-exact; zoo AR-head idiom) |
| **depth_embeddings** ×8 | per-CB SharedEmbedding: `embed_raw` conditions next CB, `get_logits` samples | host |
| **audio_embedding** | 8×2049 SharedEmbedding, sum over 8 offset CBs → feedback embed | host |
| **detokenizer** | FusedEmbedding + 8L LFM2 (5 conv + 3 sliding-attn W30) + iSTFT | **Core AI backbone (GPU) + host iSTFT** |

The host (Swift/Python) assembles the mixed text/audio prompt embeddings, runs the sequential generation
loop, does the 8-step depthformer argmax loop, and the iSTFT — the VoxCPM/dots AR-orchestration idiom.

## Milestone A — ASR / audio-understanding (audio-in → text)

The backbone can't gather its own embeddings (the prompt is mixed text + encoder audio), so it needs an
**embeds-input** decode graph. The shipped `lfm2.py` overlay gained `Lfm2Model.forward_stateful_embeds`
(copy of `forward_stateful` with `h = inputs_embeds`) + `Lfm2EmbedsForCausalLMStateful` — non-invasive
(the shipped `input_ids` path is untouched). Weights load from the audio checkpoint's `lfm.`-prefixed
tensors (arch == shipped LFM2.5-1.2B; weights == LFM2-1.2B base).

**Host prompt assembly:** `in_emb[text_pos] = lfm.embed_tokens(text_ids)`, `in_emb[audio_pos] = encoder
audio_emb`, keyed by `modality_flag` (TEXT=1, AUDIO_IN=2). Reconstructing `in_emb` from parts matches the
oracle's own assembled embeds exactly.

**Result:** on-engine (Mac GPU, AOT) greedy ASR = **byte-exact 14/14 tokens** vs `liquid_audio`, for both
the isolated-graph path (oracle embeds) and the true end-to-end path (Core AI encoder audio_emb).

## Milestone B — TTS (text → audio)

Per audio frame: LFM hidden [2048] → `depth_linear` → [8,1024] → an 8-step depthformer AR loop
(`cur_i = depth_in[i] + embed_raw(prev_code)`, CB0 cond = 0; greedy argmax per CB) → 8-CB tokens → the
frame is embedded back into the backbone via `audio_embedding(tokens + offsets).sum(0)`. Sequential mode:
emit text until `AUDIO_START(128)`, then audio frames until `AUDIO_EOS(2048)`.

**Result:** fully on-engine TTS — LFM backbone (AOT) + detokenizer (GPU) both on Core AI, only the tiny
depthformer/audio_embedding on host — reproduces the oracle greedy codes **64/64 exact**, produces audio.
Eager (pure-torch) reference = 96/96 codes exact + detok wav **SNR 113 dB**.

## Gotchas (the expensive lessons)

- **Depthformer RoPE is θ=1e6, adjacent-pair (interleaved-complex), NOT θ=1e4 rotate-half.** The oracle
  `MHA` defaults `theta=1_000_000` and uses `view_as_complex(rearrange('(D two) -> D two'))` — pairs are
  *adjacent* (x0,x1),(x2,x3)…, unlike the LFM2 backbone's split-half `rotate_half`. Re-express in real
  arithmetic: `o[2i]=x[2i]cos-x[2i+1]sin; o[2i+1]=x[2i]sin+x[2i+1]cos`. (Complex ops don't export.)
- **`SharedEmbedding` has two paths:** `embed(tok)` = raw `embedding(tok)` (NO norm) conditions the next
  codebook; `get_logits(h)` = `to_logits(embedding_norm(h))` (norm THEN project) produces the logits.
- **The coreai SDPA COMPOSITE crashes MPS lowering over a large PREFILL query block.** The detok backbone
  (S=384 prefill) with `SDPA(is_causal, window_size)` aborts at GPU load (`AICode→MPS lowering failed`) and
  AOT (`libODIECompiler` NSException) — window=0 full-causal crashes too, so it's not the sliding window.
  **Fix = raw matmul-softmax + an explicit additive mask** (the FastConformer-encoder recipe): `(q@kᵀ)*scale
  + mask → softmax → @v`, GQA via `repeat_interleave`. Then GPU JIT is cos 1.000000 / 68 dB fp16, no AOT
  needed. The composite is fine for S=1 *decode* (Milestone A); the blow-up is large query-length attention.
- **Mac-GPU decode of the LFM backbone needs AOT `--expect-frequent-reshapes`.** The plain `.aimodel` via
  raw `fn()` re-JIT-specializes per position length → ANE-compiler thrash (multi-GB scratch, hang). AOT-
  compile (`coreai-build compile --platform macOS --preferred-compute gpu --expect-frequent-reshapes
  --architecture h16c`), then load the `.aimodelc` with `SpecializationOptions.default()` (NOT `.gpu` →
  re-JIT wedge). `cpu_only` can't specialize the graph either.
- **The detok backbone is the same LFM2 hybrid arch** (conv + attn), so its config maps
  `sliding_attention → full_attention` and the window is applied by the mask, not the layer type.
- **Cryptic CoreAIRuntime errors are often DISK-FULL, not bugs.** `Indexing.swift: interleave must have rank
  (1)` (converter) and `No space left on device` (runtime) are temp-write failures — check `df -h
  /System/Volumes/Data` + the `com.apple.MetalPerformanceShadersGraph` scratch dir first.

## The detokenizer output → wav

Backbone output → `Linear(512, 1282)` → split into `log_abs`[641] + `angle`[641] → `polar(exp(log_abs),
angle)` → custom **"same"-padded iSTFT** (n_fft 1280 / hop 320, Vocos fold-based, `torch.istft` can't do
"same" padding) → 24 kHz. 90 240 samples = 47 frames × 6 (nearest-upsample 12.5→75 Hz) × 320 hop. The
`exp(log_abs)` amplifies errors, so the fp16 detok is gated on the log-domain (68 dB spec → 53 dB wav).

## Files

`coreai-models-community/conversion/lfm_audio/`: `export_encoder_adapter.py` (encoder), `export_lfm2_embeds_decode.py`
(LFM embeds decode lib) + `export_worker.py`/`export_lfm_hidden_worker.py` (logits / hidden export workers) +
`aot_gate.py` (ASR engine gate), `depthformer.py` (depth_linear + depthformer + heads + audio_embedding),
`detokenizer.py` (FusedEmbedding + backbone + iSTFT) + `export_detok_worker.py`. Gates: `_depthformer_gate.py`,
`_detok_gate.py`, `_detok_engine_gate.py`, `_tts_e2e_eager.py`, `_tts_e2e_engine.py`. Oracle parity refs:
`_lfmaudio_oracle/` (asr_ref, asr_pathref, tts_dump, tts_greedy.wav). Overlay: `Lfm2Model.forward_stateful_embeds`
+ `Lfm2EmbedsForCausalLMStateful` in `coreai-models/.../macos/lfm2.py` (mirror to `conversion/overlay/files/…`
at ship time).
