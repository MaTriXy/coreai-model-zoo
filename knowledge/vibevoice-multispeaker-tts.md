# Next-token-diffusion TTS on Core AI (VibeVoice-Realtime-0.5B)

Porting notes from the zoo's first **multi-speaker / dialogue** TTS
([`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B), MIT).
Card: [`zoo/vibevoice.md`](../zoo/vibevoice.md). Recipe: [`conversion/vibevoice`](../conversion/vibevoice).

## Export boundary

The model is a **dual LM + diffusion head + VAE decoder**, which maps 1:1 onto the VoxCPM / dots.tts
scaffolding already in the zoo:

| module | params | graph | note |
|---|---|---|---|
| `language_model` (Qwen2.5, h896) | 196 M | q=1 decode, stateful KV, cache 512 | 4 layers, `norm = Identity` |
| `tts_language_model` | 434 M | same | 20 layers; **one graph serves the positive and the CFG-negative stream** |
| `prediction_head` | 42 M | `(noisy[2,64], t[2]) → [2,64]` | the 2 is cond+uncond, not a batch of frames |
| `acoustic_tokenizer.decoder` | 344 M | `latents[1,64,T] → audio[1,1,3200·T]` | whole-sequence == streaming (cos 1.0) |
| `tts_eos_classifier` | 0.8 M | host | 2 small matmuls, not worth a graph |

The acoustic **encoder** is never exported: reference voices ship as pre-computed prefill KV in the
upstream `.pt` presets. That removes ~344 M params and the entire speaker-encoding path from the port.

Host keeps: the AR loop, the DDPM/DPMSolver++ sampler, CFG mixing, the token-embedding lookup, the EOS
decision, and the connector feedback.

## The lessons

**1. fp16 is not a choice here.** int8 LMs pass a static decode gate (cos 0.999+) and still destroy the
model: in the *speech feedback loop* — where each latent becomes the next input embedding — errors
compound to latent min cos **0.187** with early EOS. int8 main LM alone is 0.977, also not shippable.
Quantization gates must be run **in the loop**, not per-step. (Same conclusion VoxCPM and dots reached
for their continuous heads; this is the third data point.)

**2. The diffusion head is fp16-sensitive in the *reference*, not in Core AI.** A pure-torch fp16 head
collapses to cos 0.79 because the RMSNorm/adaLN reductions overflow; Core AI keeps those reductions in
fp32, so the engine graph is cos 0.999999. If you compare "fp16 engine vs fp16 torch" you will chase a
bug that does not exist — **the host DDPM reference must run fp32.**

**3. Multi-speaker needs no model surgery.** The differentiator is a host loop: split the script on
`Speaker N:`, generate each turn from its own preset with fresh noise, concatenate. No multi-speaker
prefill, no encoder. Which also means it pairs directly with the zoo's Sortformer diarizer for a
closed **generate → diarize** loop.

**4. Do not pass `expectFrequentReshapes` on iOS.** All five graphs are fixed-shape; asking for the
hint makes the runtime discard the AOT specialization and compile on device → SIGSEGV inside the
MPSGraph AICode compiler, with no error text. Details and the crash signature:
[`aot-and-specialization.md`](aot-and-specialization.md).

**5. The golden-only-N-frames trap.** The decoder's export shape `T` is fixed, but the oracle capture
only holds 30 latents. Exporting the T=64 bundle the device wants therefore needs the reference tiled
out to 64 frames — the decoder is causal, so the first 30 output frames are still gate-able against
the golden (cos 1.000005). Clamping `T` to the golden length instead (the original `min(tframes, N)`)
silently produces the *wrong bundle* and an on-device shape mismatch.

## Numbers

| gate | result |
|---|---|
| head / connector, engine vs oracle | cos 0.999999 / 1.000000 |
| decoder T=30 / T=64 vs non-stream golden | cos 1.000004 / 1.000005 |
| main LM / tts LM decode, engine vs torch | cos 0.999999 / 0.999996 |
| Python E2E (5 engines) vs upstream streamed wav | latent min 0.999198, wav 0.999479 |
| iPhone 17 Pro (h18p AOT, GPU) vs golden | cos 0.998308 |

iPhone 17 Pro: 6 graph loads in 2.6 s warm; 24 latents / 3.20 s of audio in 2.3 s = **10.6 tok/s
≈ 1.4× real-time**, all-fp16 (~1.4 GB resident across the five graphs).
