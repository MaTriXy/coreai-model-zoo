# Official-recipe models, pre-converted

Apple's [coreai-models](https://github.com/apple/coreai-models) ships export recipes
but no artifacts. These are **unmodified official-recipe conversions**, hosted on
Hugging Face with hashes, the exact export environment, and measured performance on
every card. Distinct from the zoo's community ports (which live in [`zoo/`](../zoo/)
and may use engine patches): everything here runs on the stock runtime.

| Model | Bundles | M4 Max decode (tok/s) | Download |
|---|---|---:|---|
| gpt-oss-20B (MoE) | macOS (MXFP4, 13 GB) | 78.1 | [HF](https://huggingface.co/mlboydaisuke/gpt-oss-20b-CoreAI-official) |
| Qwen3 0.6B | macOS · **macOS-26 artifact (1,121 tok/s)** · iOS | 484 | [HF](https://huggingface.co/mlboydaisuke/qwen3-0.6b-CoreAI-official) |
| Qwen3 4B | macOS · iOS | 145.4 | [HF](https://huggingface.co/mlboydaisuke/qwen3-4b-CoreAI-official) |
| Qwen3 8B | macOS | 94.1 | [HF](https://huggingface.co/mlboydaisuke/qwen3-8b-CoreAI-official) |
| Gemma 3 4B IT | macOS | 141.5 | [HF](https://huggingface.co/mlboydaisuke/gemma-3-4b-it-CoreAI-official) |
| Gemma 3 12B IT | macOS | 55.0 | [HF](https://huggingface.co/mlboydaisuke/gemma-3-12b-it-CoreAI-official) |
| Mistral 7B v0.3 | macOS | 101.7 | [HF](https://huggingface.co/mlboydaisuke/mistral-7b-v0.3-CoreAI-official) |

Protocol: Apple's official `llm-benchmark`, 512 prompt / 1,024 generated / 5 trials,
greedy. Full tables (prefill, load times, memory, iPhone 17 Pro, MLX comparison):
[`knowledge/apple-models-bench.md`](../knowledge/apple-models-bench.md) · raw data:
[apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench).

## Image generation

Diffusion bundles from Apple's official `coreai.diffusion.export` recipe, on the stock
`CoreAIDiffusionPipeline` runtime — **no model-code port** (image generation is already
supported by the Apple stack).

| Model | Bundles | M4 Max @ 4 steps | Download |
|---|---|---:|---|
| FLUX.2 klein 4B (text→image) | **macOS** (1024, int4-per-block, ~4 GB) | 1024 ≈ 17.4 s | [HF](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) |

4-step distilled (guidance 1.0, discrete-flow scheduler). Run it from the
[CoreAIImageGen app](../apps/CoreAIImageGen/) (macOS) or the `diffusion-runner` CLI.
**macOS-only**: at 4B the peak footprint (the text encoder stays resident through the
transformer) runs ~0.4 GB over a 12 GB iPhone's per-process limit even AOT-compiled, so the
iOS bundle is not shipped. Smaller diffusion bundles (Stable Diffusion 0.9B) run fine on iOS.

## Segmentation

Promptable segmentation from Apple's official `models/<name>/export.py` recipes, on the stock
`CoreAIImageSegmenter` runtime — no model-code port.

| Model | Bundle | Prompt | M4 Max (warm) | Download |
|---|---|---|---:|---|
| SAM 3 (Meta) | macOS · iOS (float16, ~1.7 GB) | **text** (open-vocabulary) | ~0.55 s / image | [HF](https://huggingface.co/mlboydaisuke/sam3-CoreAI-official) |

Give it an image and a phrase (`cat`, `the red car`) → instance masks + boxes + scores, no
fixed class list. float16 matches the float32 reference (top scores Δ≤1e-4, masks within 1 px).
Run it from the [CoreAISegment app](../apps/CoreAISegment/) (macOS + iOS) or
`swift run -c release image-segmenter --model <bundle> --prompt "cat" --image cats.jpg`. SAM 3
is gated upstream and ships under Meta's SAM License (see the card); the converted bundle lets
you run it without the gated checkpoint.

## Speech-to-text

Whisper ASR from Apple's official `models/whisper/export.py` recipe, on the stock runtime.

| Model | Bundle | M4 Max (warm) | Download |
|---|---|---:|---|
| Whisper large-v3-turbo (809M) | macOS · iOS (float16, ~1.5 GB) | ~0.18 s/token | [HF](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) |

Transcribes on-device, token-for-token identical to the PyTorch reference. The stock recipe
traces a **single** decode step (`decoder_input_ids [1,1]`, no KV cache) — which can't be
driven autoregressively — so this bundle is the same recipe traced at a **fixed 128-token
decoder window**: pad the buffer, read logits at the real last position (causal attention
ignores the padding), constant shape so MPSGraph compiles once. **Card with the runnable
snippet + integration checklist: [`models/whisper-large-v3-turbo/README.md`](../models/whisper-large-v3-turbo/README.md)**;
run it from the [Transcribe example ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe)
(GUI + CLI) or the [CoreAITranscribe app](../apps/CoreAITranscribe/) (macOS + iOS, file or mic).
The bundle ships the tokenizer + the mel filterbank for the Swift log-mel frontend.

## Why artifacts and not just recipes?

The same export command can produce a 2.2× slower artifact across an OS upgrade
(macOS 26 → 27β changed the quantization lowering — op-level forensics
[here](https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/methodology/coreai-export-lowering.md)).
An `.aimodel` is a build artifact: these are the exact, hash-stamped bundles behind
the published numbers. The Qwen3-0.6B repo includes the macOS-26-era artifact that
current toolchains can no longer reproduce.

## Run them

- CLI (LLM): `swift run -c release llm-runner --model <bundle-dir> --prompt "Hello"`
- CLI (diffusion): `swift run -c release diffusion-runner --model <bundle-dir> --prompt "a cat"`
- CLI (segmentation): `swift run -c release image-segmenter --model <bundle-dir> --prompt "cat" --image cats.jpg`
- Apps: [CoreAIChatMac](https://github.com/john-rocky/coreai-samples) (chat) ·
  [`apps/CoreAIImageGen`](../apps/CoreAIImageGen/) (image generation, iOS + macOS) ·
  [`apps/CoreAISegment`](../apps/CoreAISegment/) (text-prompt segmentation, iOS + macOS) ·
  [`apps/CoreAITranscribe`](../apps/CoreAITranscribe/) (speech-to-text, iOS + macOS)
- iOS bundles need AOT compilation first — see each model card.

Licenses: bundles inherit their upstream model licenses (Apache-2.0 for Qwen /
Mistral / gpt-oss / FLUX.2 klein / Whisper; Gemma Terms of Use for Gemma; Meta SAM License
for SAM 3 — see the cards).
