# VoxCPM-0.5B on Core AI — diffusion TTS, on-device (iPhone + Mac)

[OpenBMB VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) ported to Apple's Core AI. It is the zoo's second TTS (after Kokoro) and the first **diffusion** one. Apache-2.0. Conversion in `conversion/voxcpm/`; Swift host in [CoreAIKit](https://github.com/john-rocky/coreai-kit) (`VoxCPMTTS`); demo = `apps/coreai-audio` "Voice" tab. Weights: [🤗 mlboydaisuke/VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI).

## Architecture

VoxCPM is not a vocoder TTS. It generates 16 kHz audio through a continuous, token-rate (12.5 Hz) loop:

- **base_lm** — MiniCPM4 dense backbone (24L, hidden 1024, GQA 16q/2kv, head_dim 64). `use_mup=False` ⇒ scale_emb 1.0, no scale_depth; short==long RoPE factors ⇒ fixed RoPE. So it's ≈ a vanilla dense LM that takes `inputs_embeds` and returns hidden (no LM head in the loop).
- **residual_lm** — same backbone, 6L, fed the base hidden.
- **feat_decoder** — LocDiT (4L bidirectional) wrapped in a UnifiedCFM euler loop (10 steps, CFG 2.0, cfg-zero-star, sway schedule). The only stochastic input is the initial noise `z`. Predicts one audio patch `[1,64,2]` per frame.
- **feat_encoder** — LocEnc (4L) + projection: turns the predicted patch back into the per-frame embedding fed to base_lm.
- **AudioVAE** — DAC-style causal-conv decoder, 640× upsample, deterministic.
- **FSQ** — scalar quantization bottleneck between base and residual (`tanh → round(h·9)/9`).

Per frame: `dit = lm_to_dit(lm_h) + res_to_dit(res_h)` → `feat_decoder(dit, prev_patch, z)` → `feat_encoder` → stop-head check → `base_lm.decode` → `FSQ` → `residual_lm.decode`. Latents → AudioVAE → wav.

## Port lessons

- **Prefill via the decode bundle (no prefill bundle).** The q=1 decode graph is causal with an explicit `arange<=pos` mask, so looping it over the text embeds (pos 0..T-1) is *bit-identical* to a batched prefill. This drops the two baked-length prefill bundles (~856 MB) and makes text length unbounded — no per-length re-export. Ship is 5 bundles.
- **Five bundles, host-side glue.** base/res decode (stateful KV) + feat_decoder + feat_encoder + vocoder run on the engine; the small pieces (token embedding, lm/res→dit projections, FSQ, stop head) run host-side via Accelerate. enc→lm projection is folded into the feat_encoder bundle.
- **Baked RoPE.** Bake cos/sin constants + manual `rotate_half` and drop `rope` + `scaled_dot_product_attention` from the export's externalize specs (engine-native externalized RoPE mishandles baked position ids; the explicit causal-buffer mask can't go through engine SDPA).
- **int8 the LM only; keep the diffusion + VAE full precision.** Weight-only symmetric per-channel int8 on base/res (685→343 MB, 171→86 MB). The feat_decoder/feat_encoder/AudioVAE stay fp16: the continuous-hidden feedback loop amplifies quantization error far more than a token-argmax LLM does, so quantizing the diffusion path degrades output. This is exactly the split mlx-community uses for VoxCPM2 (`targets: [base_lm, residual_lm]`). Per-step int8 cos > 0.999 vs the fp32 reference; whole-utterance output is natural speech.
  - Note: reproduce-oracle (trajectory fidelity to a fixed-noise fp32 run) is the *wrong* gate for a quantized stochastic AR model — small perturbations diverge the trajectory by design. Gate the LM per-step and judge the diffusion/VAE by listening.

## On-device

- macOS: JIT `.aimodel` (GPU). iPhone: AOT `.aimodelc` (`coreai-build compile --platform iOS --preferred-compute gpu --architecture h18p`) — all 5 bundles specialize cleanly (no `failedToSpecialize`). int8 LM brings the iOS `.aimodelc` set to ~1.0 GB.
- iPhone 17 Pro: real-time, fits with the increased-memory entitlement. M-series Mac: RTF ~0.6 (fp16) / ~0.86 (int8 weight-only adds GPU dequant overhead — the int8 win is size/memory, not speed).

Plain TTS today; VoxCPM's voice-cloning branch (prompt VAE-encode + prompt prefill) is a follow-on.

## Latency: streaming + batched-prefill bundle

The "too slow to generate" complaint is about *time-to-first-audio*, not throughput. Two changes fix it with zero quality change (same int8 model):

- **Streaming.** `synthesizeStreaming` vocodes and emits every `vaeFrames/patch = 6` frames (one 12-column window = 0.48 s) as the AR loop runs, instead of returning only after the whole clip. AudioVAE is causal and the windows are 12-column-aligned from 0, so the streamed output is byte-identical to batched `synthesize`. Playback starts after the first chunk.
- **Batched q=32 prefill bundle.** The bit-identical prefill-via-decode loop (above) re-reads all backbone weights once *per text token* — costly on the bandwidth-bound A19. A q=32 batched-prefill bundle seeds the KV cache in a single pass; `StatefulGraphModel.adoptState` copies that KV straight into the decode state (no fp16 round-trip). The int8 ship has a matching int8 prefill bundle (base 343 MB / res 86 MB).

iPhone 17 Pro, int8, same model — OLD (no prefill bundle, non-streaming) vs NEW: synthesis emits its first chunk at **~0.43 s** (vs ~4.2 s for the whole clip) and runs at **RTF ~0.9** (just below real-time). Because RTF sits close to 1.0, the app holds a small **pre-roll jitter buffer (~2 chunks ≈ 1 s)** before starting playback, so a momentary generation slowdown can't starve the player (audible crackle/underrun). Net *perceived* time-to-first-audio is **~0.9 s** — still ~5× better than the ~4 s whole-clip wait — with smooth, gapless playback.

Throughput note: on the bandwidth-bound A19 the backbone q=1 GEMV is the floor, so int8 and fp16 tie on RTF — the win is first-audio + streaming, not raw tokens/s. (On M-series the picture differs — fp16 and an fp16-on-ANE backbone are faster — but those gains don't transfer to iOS: the ANE backbone path is uncompilable for h18p, and a custom Metal matvec lost to the engine's own GEMV.) A larger real-throughput win needs few-step / CFG distillation of the diffusion, which is a quality trade-off.

# VoxCPM2 (2B, 48 kHz) on Core AI

[OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — the 2B, 48 kHz successor — is the zoo's first **2B on-device TTS**. Same five-bundle + host-glue shape as the 0.5B, scaled up and re-architected in a few places. Swift host = `VoxCPM2TTS`; demo = `apps/coreai-audio` "Voice 2B" tab. Weights: [🤗 mlboydaisuke/VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI). Conversion = `conversion/voxcpm/*_v2.py` (+ `pipeline_v2.py`, a self-contained generator that loads only from the raw checkpoint = the Swift-host reference).

## What changed vs 0.5B

- **Scale.** base_lm 28L, residual_lm 8L (`no_rope: true`), hidden 2048, head_dim 128 (= `kv_channels`, **not** hidden/heads), 16q/2kv, ffn 6144. feat_decoder LocDiT **12L**, feat_encoder LocEnc **12L** (1024h / 4096ffn / 128hd). patch 4 (was 2), FSQ latent 512 (was 256).
- **Dataflow deltas — the easy place to get a silently-wrong port.** (1) `mu = concat(lm_to_dit(lm_h), res_to_dit(res_h))` → 2048, reshaped to **two** 1024-dim DiT tokens (0.5B *added* them into one). (2) In LocDiT the timestep token is **separate** from mu: `seq = [mu(2), t(1), cond(P), x(P)]`. (3) The residual-LM input is `fusion_concat_proj(cat(A, B))` where A=`fsq(lm_h)`|prefill-base-out, B=`curr_embed`|0 (0.5B just added lm_h + curr). In zero-shot prefill the residual input is `fusion(cat(base_out, 0))` per position — asymmetric with the loop.
- **AudioVAE v2 = 48 kHz.** Decoder rates `[8,6,5,2,2,2]` = **1920×** (25 Hz latent → 48 kHz), **depthwise** convs, and a `SampleRateConditionLayer` (scale_bias) before every upsample block. The output rate is fixed (48000 → `bucketize([20000,30000,40000]) = 3`), so the per-channel scale/bias embeddings are **baked into constant buffers at load** — no embedding lookup in the graph. weight_norm folded as usual.

## Verification (the oracle)

Reassemble the official VoxCPM2 source into its expected package layout (`_ref_v2/voxref/`, with a no-op LoRA stub) so the full `VoxCPM2Model` instantiates on CPU = a true oracle. Then: per-component gates (backbone / feat_decoder / feat_encoder) **cos 1.0** vs the official modules loaded from the real checkpoint; an e2e gate replays the official CFM noise through my overlays and matches latents (cos 0.997) + full-chain 48 kHz **magspec 0.996**; every exported bundle engine-gated **cos ≥ 0.9999**.

## On-device (iPhone 17 Pro)

- **Depthwise grouped Conv1d + ConvTranspose + baked SR-cond AOT-compile clean for h18p** (0 `failedToSpecialize`) — the one thing the 0.5B never exercised; it just works.
- int8 backbone (base 1.2 GB / res 360 MB each for decode + prefill) + fp16 feat_decoder/encoder/vocoder + glue ≈ **4.9 GB** `.aimodelc`, fits with the increased-memory entitlement (no jetsam). A `MINIMAL` set (drop the two prefill bundles → prefill-via-decode, bit-exact) is 3.3 GB.
- int8 + int8-prefill + streaming: **first-audio 0.65 s, RTF ~1.19, 48 kHz**. The 2B is ~4× the 0.5B, so RTF sits *above* 1.0 — streaming falls behind by ~10 %, so the app needs a **3-chunk pre-roll** (not 2) to avoid crackle. Below-real-time needs diffusion distillation (quality trade). Mac RTF (~0.97) does not transfer (A19 is bandwidth-bound).

## Gotcha

Uploading the large Core AI bundles to the Hub: the **xet** transfer backend panics mid-file on the multi-GB `.mlirb`/`.aimodelc` blobs (`assertion left == right ... not fully completed`). Set **`HF_HUB_DISABLE_XET=1`** (classic LFS path); `upload_folder` is idempotent, so a re-run skips completed files and finishes clean.
