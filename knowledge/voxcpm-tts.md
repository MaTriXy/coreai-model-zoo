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
