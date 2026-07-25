# VibeVoice-Realtime-0.5B → Core AI — SHIP NOTES

zoo's **first multi-speaker / dialogue (podcast-style) TTS**. **Shipped 2026-07-23** (HF + zoo card +
README row + knowledge doc); the live card is `models/vibevoice/README.md` and the ladder is in `KICKOFF.md`.
What follows is the drafting state that preceded the push, kept for the recipe.

## Status (2026-07-07)
- ✅ All 5 graphs ported + engine-gated ≥0.9999 (see KICKOFF.md step 3).
- ✅ Python E2E (all-fp16 engines) reproduces the upstream streamed wav at **cos 0.999473**.
- ✅ Multi-speaker dialogue demo (host turn-switching, `host_multispeaker.py`).
- ✅ AOT-compiled for iPhone h18p (all 5, `artifacts_ios/`).
- ✅ Swift host: `ondevice/VibeVoiceRunner` (Mac-validated cos 0.999) + `coreai-audio` self-test
  (`VibeVoiceSelfTest.swift`, app builds+signs for iOS).
- ⛔ **Device gate blocked**: iPhone install `error 4016` = device LOCKED. Unlock, then:
  `xcrun devicectl device install app --device A6F3E849-1947-5202-9AD1-9C881CA58EEF <coreai-audio.app>`
  → `conversion/vibevoice/sideload_ios.sh A6F3E849-1947-5202-9AD1-9C881CA58EEF`
  → launch with `VIBEVOICE_SELFTEST=1` → console `[VV] gate vs golden: cos=… -> PASS`.

## zoo README row (audio/speech table, after VoxCPM2 / near Stable Audio)
```
| **VibeVoice-Realtime-0.5B** (text → **multi-speaker dialogue** — the zoo's first multi-speaker / podcast-style TTS: Qwen2.5 dual-LM (4L context + 20L speech) + next-token diffusion head (DPMSolver++ v-pred) + causal-conv acoustic VAE, 24 kHz; host turn-switching for N-speaker conversations; pairs with Streaming Sortformer for a **generate → diarize** loop; iPhone + Mac, all-fp16) | [🤗 VibeVoice-Realtime-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI) | [coreai-audio](../../apps/coreai-audio) | MIT |
```
Do NOT claim "first on-device" (obscure CoreML/GGUF ports exist). Frame = "zoo's first
multi-speaker/dialogue TTS, app-integrated, paired with Sortformer diarization".

## HF model card (mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI, MIT)
- Base: microsoft/VibeVoice-Realtime-0.5B (MIT). EN/ZH.
- Architecture: dual Qwen2.5 LM (4-layer text context, norm=Identity + 20-layer speech trunk) →
  per-frame **next-token diffusion** (4-layer adaLN head, DDPM cosine, v-prediction, DPMSolver++
  5-step) → causal-conv acoustic VAE decoder (7.5 Hz latent → 24 kHz, upsampling ×3200/frame).
- Bundles (all fp16): main LM 114M, tts LM 569M, diffusion head 80M, connector 1.7M, decoder 656M.
  iOS = AOT `.aimodelc` (h18p). Voice = pre-computed prefill `.pt` presets (25, EN/ZH/…); the
  acoustic **encoder** is not shipped (references are pre-encoded into the presets).
- Quality: engine E2E cos 0.9995 vs upstream. **fp16 LMs required** — int8 diverges in the speech
  feedback loop. Multi-speaker = host turn-switching (one preset per "Speaker N:" turn).

## Conversion recipe (reproduce)
```
env = coreai-models/.venv (export) + conversion/vibevoice/_oracle/.venv (torch oracle, HF_HUB_DISABLE_XET=1)
oracle.py → oracle_ref.npz + oracle.wav                 # golden per-stage fixtures (5-step DDPM)
oracle_decode_ref.py / oracle_lm_ref.py                 # non-stream + fresh-prefill goldens
export_diffusion_head.py / export_decoder.py --tframes 30 / export_backbone.py --mode fp16
host_e2e.py --backend engine --lm-mode fp16             # E2E gate (wav cos 0.9995)
host_multispeaker.py                                    # multi-speaker dialogue demo
xcrun coreai-build compile <aimodel> --platform iOS --architecture h18p --preferred-compute gpu   # AOT
pack_device_inputs.py → device_bundle/                  # compact on-device host inputs
```

## Ship checklist (when approved)
1. Seed kit ModelStore / HF: upload the 5 macOS `.aimodel` + iOS `.aimodelc` + voice presets + card.
2. zoo README row (above) + `models/vibevoice/README.md` page + conversion recipe into `conversion/`.
3. knowledge doc (dual-LM next-token-diffusion TTS; fp16-feedback lesson) → `community/knowledge/`.
4. coreai-audio: promote the self-test into a proper "Dialogue" tab (VibeVoiceView) if desired.
5. X post: understated (feature/technique-led, subtle zoo link). Pair-with-Sortformer angle.
```
```
