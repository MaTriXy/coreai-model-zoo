# North-Micro-Vision-Instruct (2.4B, vision-language) — Core AI

**A multilingual native-resolution VLM that runs on a phone at 24/24 token-exact.** A Core AI
port of [`CohereLabs/North-Micro-Vision-Instruct`](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct):
image + text → text on the [pipelined-engine fast path](../../knowledge/pipelined-engine.md),
Apache-2.0, eleven languages including Japanese.

Architecture (`model_type: cohere_compass`): a **400M vision tower custom-trained from SigLIP2
SO400M** — which turns out to be a **Qwen3-VL visual encoder** in every structural respect
(fused `attn.qkv`, `merger` + three `deepstack_merger_list` stages, an interpolated `pos_embed`
table, `patch_embed.proj [d,3,2,16,16]`) at SigLIP2's dimensions — and a **2B Cohere decoder**
that is not a Llama with different numbers:

- **parallel block** — one `input_layernorm` per layer, attention and MLP both read it and
  their outputs are summed into the residual;
- **Cohere LayerNorm** — the mean *is* subtracted, and there is no bias;
- **`SSSF × 7` layer types** — the 21 sliding layers carry interleaved M-RoPE (sections
  [24,20,20], θ 5e4) inside a 4096 window, and the 7 full-attention layers have **no positional
  encoding at all**;
- **`logit_scale` 0.25** and a 262 144-entry embedding **tied** to the head — 1.07 GB of fp16
  that no linear quantization touches.

Only the decoder was authored ([`models/macos/cohere_compass.py`](../../conversion/overlay/));
the tower is the zoo's existing `Qwen3VLVisionEncoder` with a config shim.
[`knowledge/north-micro-vision-port.md`](../../knowledge/north-micro-vision-port.md).

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/North-Micro-Vision-CoreAI](https://huggingface.co/mlboydaisuke/North-Micro-Vision-CoreAI)** —
`gpu-pipelined/` (vision fp16 + int8lin decoder + text core) and `ios-h18p/` AOT variants of the
shipped pair, device-gated on an iPhone 17 Pro. Apache-2.0.

## Numbers

**M4 Max**, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`:

| artifact | size | measured | numerics |
|---|---:|---|---|
| **vision fp16 (ship)** | **1.0 GB** | **83.4 ms**/image (median of 12) | `image_embeds` cos **0.999996** vs fp32 HF |
| **decoder int8lin (ship)** | **2.4 GB** | text core: **145.3 prompt / 118.6 decode tok/s** | suite **9/9 cases, 338/338 tokens** |
| decoder int4lin | 1.8 GB | not benchmarked | suite **0/9** — **NO-GO**, see below |

**iPhone 17 Pro** (`ios-h18p`, AOT `--architecture h18p`, PipelinedBench):

| bundle | prefill | decode | numerics |
|---|---:|---:|---|
| **`decode_int8lin` (image bound)** | **21.5** | **18.2** | nat 16/16 + **image oracle 24/24 vs fp32** |
| `decode_int8lin`, `PB_G=1024` | 21.8 | 18.2 | 16/16 + 24/24, no collapse |

Engine ready in 4.6 s cold. The `PB_G=1024` row is mandatory, not extra: the iOS compiler
miscompiles KV specializations at seq ≥ 2048 and `g ≤ 256` cannot see it.

**This bundle's AOT `resources.bin` is 2.39 GiB** — past where this repo's own note put the iOS
load wall (2 GiB, inferred from a 1.96 ✅ / 3.92 ❌ bracket). It loads. Two models on the same
day moved that bracket: LFM2.5-VL-3B at 2.03 GiB and this at 2.39 GiB. **Measure the artifact,
then ask the phone; do not infer a verdict from a bracket with a 2 GiB gap in it.**

## Gates

- **fp32 torch ladder** (`_smoke/test_northmv_torch_ladder.py`) — cos **1.000000** at every
  vision seam and on `logits_last`, token-exact greedy, at the fixture's own 30×40 native grid
  *and* at the 16×16 merged grid the bundle bakes.
- **host preprocessing** (`_smoke/northmv_preprocess.py`) — patchify + normalize **bit-exact**
  against the processor; the antialiased BICUBIC resize agrees with Pillow to cos 0.999997.
- **engine** (`_smoke/test_northmv_aimodel_gate.py`) — the exported bundles chained the way an
  app runs them: vision cos 0.999996, decoder logits cos 0.999968, 46/46 token-exact.
- **compression** (`_smoke/test_northmv_suite_gate.py`) — 9 image × prompt cases, **9/9**.

Cosines are computed in float64; in float32 the reduction over ~10⁶ elements prints
`cos 1.000088` for two identical tensors.

The oracle needs **transformers git main** (5.16.0.dev0 at the time of writing): 5.15.0 does not
know `cohere_compass` at all, which at least fails loudly.

## What compression costs

**int8 costs this model nothing measurable**: 9/9 cases and 338/338 tokens against fp32, and
24/24 on device. There is no fp16 baseline row because a perfect score does not need one — the
baseline exists to explain a *gap*.

**int4 craters, and not quietly**: 0/9, 23.7 % of tokens, with the failure modes that name
themselves — a sentence boundary lost and instruction boilerplate leaking in
(*"…possibly a couch or a blanket.**Answer: Cats.I apologize, but I cannot provide a detailed
description of the image**"*), and outright repetition (*"Images of cats sleeping on a couch are
shown.Images of cats sleeping on a couch are shown."*). It is not published.

That is the third int4 verdict in this family of ports and they do not line up with size:
LFM2.5-VL-450M craters (0/9), LFM2.5-VL-3B does not move (7/9 = its own fp16 baseline), and this
2.4B craters. **int4 tolerance is a property of the model in front of you; read its generations.**

## How the image reaches the decoder

Identical to the Qwen3-VL rider, because the tower is that tower: the vision bundle emits
`image_embeds [256, 2048]` + `deepstack_embeds [768, 2048]` from `patches [1024, 1536]`, the
host rewrites the prompt's `<image>` ids (255031) to extension ids `V + slot`, and the decoder's
first three layers add their deepstack rows at image positions. Positions follow the same
rope-shift contract (an image consumes only max(H,W) rope positions).

The upstream processor is **native-resolution** — it kept the 640×480 fixture's own 30×40 patch
grid — and the export bakes a square 512×512 canvas instead. That stretches non-square images,
and no gate here can see the cost, because the fixed-grid oracle is captured through the same
stretch.

## Convert / verify

```bash
# vision tower + VLM decoder (the published pair)
python conversion/export_northmv_pipelined.py int8lin

# the text-core proxy: same weights, no image inputs (benchmarks)
python conversion/export_northmv_pipelined.py int8lin --text-core --skip-vision

# oracles — transformers git main ONLY (5.15.0 does not know cohere_compass)
~/code/litertlm-convert/.venv-vl0930-t515/bin/python _smoke/northmv_ref.py --resize 512x512
~/code/litertlm-convert/.venv-vl0930-t515/bin/python _smoke/northmv_suite_ref.py

# gates
python _smoke/test_northmv_torch_ladder.py --ref _smoke/north_micro_vision_instruct_ref_512x512.npz
python _smoke/northmv_preprocess.py
python _smoke/test_northmv_aimodel_gate.py --bench-vision 12
python _smoke/test_northmv_suite_gate.py --mode int8lin --show-text
```
