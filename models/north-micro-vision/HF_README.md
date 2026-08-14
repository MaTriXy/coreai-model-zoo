---
library_name: transformers
license: apache-2.0
base_model: CohereLabs/North-Micro-Vision-Instruct
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - vision-language
  - siglip2
  - cohere
language:
  - en
  - de
  - fr
  - es
  - it
  - pt
  - hi
  - ja
  - ko
  - zh
  - ar
pipeline_tag: image-text-to-text
---

# North-Micro-Vision-Instruct — Apple Core AI (`.aimodel`)

**Cohere's 2.4B multilingual VLM converted to Apple's Core AI** (the Core ML successor announced
at WWDC26), running on iOS 27 and macOS 27. On an iPhone 17 Pro it answers about an image at
**18.2 tok/s and 24/24 tokens identical to the fp32 reference** — not "close", identical.

Two bundles, run in sequence: a **400M vision tower** (custom-trained from SigLIP2-SO400M, and
structurally a Qwen3-VL visual encoder — deepstack mergers included) emitting `image_embeds
[256, 2048]` + `deepstack_embeds [768, 2048]`, and a **2B Cohere decoder** with the image tokens
spliced in as extension ids. Eleven languages, including Japanese. Apache-2.0.

> Requires iOS 27 / macOS 27 (Core AI ships with the OS). Conversion code, gates and knowledge
> base: **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | measured | numerics |
|---|---:|---|---|
| `gpu-pipelined/north_micro_vision_instruct_vision_fp16` | 1.0 GB | **83.4 ms**/image (M4 Max) | `image_embeds` cos **0.999996** vs fp32 |
| `gpu-pipelined/north_micro_vision_instruct_decode_int8lin` | 2.4 GB | — | suite **9/9 cases, 338/338 tokens** |
| `gpu-pipelined/…_decode_int8lin_textcore` | 2.4 GB | **145.3 prompt / 118.6 decode tok/s** (M4 Max) | the same weights with no image inputs |
| `ios-h18p/…_decode_int8lin` + `ios-h18p/…_vision_fp16` | 2.5 GB | **21.5 prefill / 18.2 decode tok/s** (iPhone 17 Pro) | nat 16/16 + **image oracle 24/24** |

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`. The Mac tok/s row is the text
core because `llm-runner` cannot bind an image buffer. The iPhone rows are PipelinedBench,
including a mandatory 1024-token generation (the iOS compiler miscompiles KV specializations at
seq ≥ 2048 and a 256-token run cannot see it) — clean.

**int8 costs this model nothing measurable.** **int4 is not published**: 0 of 9 cases, with a
lost sentence boundary and instruction boilerplate leaking in (*"…a blanket.**Answer: Cats.I
apologize, but I cannot provide a detailed description of the image**"*) plus flat repetition.
int4 tolerance is a property of the individual model — a sibling port at 3B took int4 for free
and a 450M one cratered — so read the generations rather than the parameter count.

## Run it

```bash
git clone https://github.com/apple/coreai-models   # + the zoo's engine patches, see below
swift build -c release --product llm-runner

COREAI_CHUNK_THRESHOLD=1 .build/release/llm-runner \
  --model gpu-pipelined/north_micro_vision_instruct_decode_int8lin_textcore \
  --prompt "The alphabet begins A, B, C," \
  --max-tokens 64 --sampling-strategy greedy \
  --inference-engine-variant coreai-pipelined --warmup off
```

The `coreai-pipelined-static-inputs` patch (which binds `image_embeds`, `deepstack_embeds` and
the two rope-shift scalars) is in the zoo under `apps/`.

For the image path the host resizes to a 512×512 canvas with an **antialiased BICUBIC** filter,
normalizes `(x/255 − 0.5)/0.5`, and patchifies into 16×16 patches in **Qwen-VL order** —
block-major over 2×2 merge groups, and `[C][T][py][px]` inside each patch with the still frame
duplicated (`patch_dim` 1536). Then it runs the tower, binds both outputs, and rewrites the
prompt's `<image>` ids (255031) to `V + slot`. Reference implementation:
[`_smoke/northmv_preprocess.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/_smoke/northmv_preprocess.py).

Note the export bakes a square grid while the upstream processor is **native-resolution** (it
keeps a 640×480 image's own 30×40 patch grid), so non-square images are stretched.

## Converting this yourself

The oracle needs **transformers git main**: the 5.15.0 release does not know `cohere_compass`
and raises on `AutoConfig`.

The decoder is where the work is, and each of these runs fine when done wrong: a **parallel
block** (one LayerNorm, attention and MLP summed into the residual), **Cohere LayerNorm** (the
mean is subtracted, no bias), `SSSF × 7` layer types where the 7 full-attention layers have **no
positional encoding at all** while the 21 sliding ones carry interleaved M-RoPE in a 4096
window, `logit_scale 0.25`, and a 262 144-entry embedding tied to the head.

The tower needed no work: it loads into the zoo's existing Qwen3-VL encoder with zero missing
keys and matches every seam at cos 1.000000.

Everything is in
[`conversion/export_northmv_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_northmv_pipelined.py)
and [`knowledge/north-micro-vision-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/north-micro-vision-port.md).

## License

Apache-2.0, carried from
[`CohereLabs/North-Micro-Vision-Instruct`](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
(revision `373bda96ac70bf89f99f7048f420cf00dc07c149`). Not affiliated with Apple or Cohere.
