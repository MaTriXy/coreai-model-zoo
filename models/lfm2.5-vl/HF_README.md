---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-VL-450M
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - lfm2
  - vision-language
  - siglip2
pipeline_tag: image-text-to-text
---

# LFM2.5-VL-450M — Apple Core AI (`.aimodel`)

**LiquidAI's LFM2.5-VL-450M converted to Apple's Core AI** (the Core ML successor announced at
WWDC26), ready to run on iOS 27 / macOS 27. Image + text → text in **658 MB** — small enough to
sit inside an app rather than be one.

Two bundles, run in sequence: a **SigLIP2-NaFlex vision tower + projector** (`patches [1024,768]
→ image_embeds [256,1024]`) and the **LFM2 conv+attention hybrid decoder** — the same decoder as
[LFM2.5-1.2B](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI), reached in this checkpoint
by a `model.language_model.` key prefix — with the image tokens spliced in through a static
`image_embeds` input. Hidden 1024, 16 layers = 10 short-conv + 6 GQA attention, vocab 65 536,
tied head. No recurrent scan, so decode is loop-free and rides Apple's `coreai-pipelined` GPU
engine with no custom kernels.

Not a thinking model (unlike the 2.6B): the generation prompt does not open `<think>`.

> Requires the iOS 27 / macOS 27 beta (Core AI ships with the OS). Conversion code, gates and
> knowledge base: **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | measured (M4 Max) | numerics |
|---|---:|---|---|
| `gpu-pipelined/lfm2_5_vl_450m_vision_fp16` | 181 MB | **18.0 ms**/image | `image_embeds` cos **0.999996** vs fp32 HF |
| `gpu-pipelined/lfm2_5_vl_450m_decode_int8lin` | 477 MB | — | 7/9 suite cases token-exact (fp16 baseline: 8/9) |
| `gpu-pipelined/lfm2_5_vl_450m_decode_int8lin_textcore` | 477 MB | **609.2 prompt / 387.2 decode tok/s** | oracle gate **PASS 16/16** |

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`.

The Mac tok/s row is the **text core** — the same decoder weights exported with no image input —
because `llm-runner` has no way to bind the VLM bundle's `image_embeds` buffer. The text core is
also a usable 350M LFM2 text model on its own.

### iPhone 17 Pro (AOT h18p, `ios-h18p/`, settled)

| bundle | prefill | decode | numerics |
|---|---:|---:|---|
| **`decode_int8lin`, image bound** | **123.2** | **112.0** | nat 16/16 + image oracle 24/24 |
| `decode_int8lin_textcore` | 122.1 | 110.6 | nat 16/16 + oracle 16/16 |
| `decode_int8lin`, g=1024 | 122.4 | 108.6 | no collapse |
| **`vision_fp16`** | — | **33.6 ms**/image | cos 0.999995 vs the same tower on Mac |

Binding the 256×1024 fp16 image buffer costs nothing per step — the VLM bundle and the text core
measure the same speed within noise. Engine ready in 0.5 s warm.

On device the image path describes the same picture as fp32 but is not token-identical: it drops
one adjective at a near-tie (*"two tabby cats … stretched out on its side"* → *"two cats … lying
on its side"*), with the tokens between the two forks identical. That is the fp16 near-tie class,
not an image-path error.

The tower's **first** encode pays ~860 ms of on-device compile; warm it with a dummy encode at
load and the user's first photo gets the 33.6 ms number instead.

## What it is good at, and what it is not

A 450M VLM answers scene-level questions — what is in the picture, where it is, which colours
dominate — and misses fine-grained geometry. That is the checkpoint, not the conversion: the
same weights on other on-device runtimes show the same split. Treat it as a caption / triage
model that fits beside an application.

The bundle bakes **one 512×512 patch grid** (32×32 patches → 2× unshuffle → 256 tokens, which is
exactly the checkpoint's own `max_image_tokens`). The upstream model is NaFlex — it picks a grid
per image and keeps the aspect ratio — so a non-square image is stretched here. That is the
price of a fixed graph, and it is the one thing to weigh before choosing this over the source
model.

`int4` is **not published**: 0 of 9 gate cases token-exact, and the failure mode is fluent drift
rather than obvious breakage — a kitchen becomes "a traditional *Italian* kitchen" where fp32
says "historical or rustic". Read generations, not loss curves, before trusting int4 on a model
this small.

## Run it

```bash
git clone https://github.com/apple/coreai-models   # + the zoo's engine patches, see below
swift build -c release --product llm-runner

# the text core (no image), to check the decoder end of the pair
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-runner \
  --model gpu-pipelined/lfm2_5_vl_450m_decode_int8lin_textcore \
  --prompt "The alphabet begins A, B, C," \
  --max-tokens 64 --sampling-strategy greedy \
  --inference-engine-variant coreai-pipelined --warmup off
```

`--warmup off` matters: default warmup submits a synthetic 256-token prefill and these bundles
are static-S=1. The engine patches (`coreai-pipelined-extra-states` for the conv state,
`coreai-pipelined-static-inputs` for `image_embeds`) are in the zoo under `apps/`.

For the image path, the host does three things: resize to 512×512 with an **antialiased**
bilinear filter (PIL/torchvision `antialias=True`, *not* a 2×2 GPU bilinear tap), normalize
`(x/255 − 0.5)/0.5`, and patchify into 16×16 patches with the **channel as the fastest axis**
(`[y][x][c]`). Then run the vision bundle, bind its output as `image_embeds`, and rewrite the
prompt's `<image>` ids (id 396) to `V + slot`. The reference implementation is
[`_smoke/lfm25vl_preprocess.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/_smoke/lfm25vl_preprocess.py).

## Converting this family yourself

The trap that costs a day: **build the oracle on transformers ≥ 5**. transformers 4.57.6 applies
the projector's LayerNorm unconditionally, while this config sets `projector_use_layernorm:
false` and ships no such weights — and `nn.LayerNorm`'s default init (weight 1, bias 0) means no
warning and no visible garbage, just a quietly different reference that would certify a wrong
port as PASS.

Two more, both readable straight off the weight shapes: `patch_embedding.weight` is `[768, 768]`
— a **Linear over pre-flattened patches**, not a Conv2d over an image — and
`position_embedding.weight` is `[256, 768]`, a 16×16 grid that is **bilinearly resized (with
antialias) to the actual patch grid**. A port written from a MiniCPM-V or Qwen-VL SigLIP recipe
gets both wrong and still produces fluent text.

Everything is in
[`conversion/export_lfm25vl_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_lfm25vl_pipelined.py)
and [`knowledge/lfm2.5-vl-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/lfm2.5-vl-port.md).

## License

LFM Open License v1.0, carried from
[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) (revision
`fc6221ca597f3315e4f82fc2df606783267b34ba`). Not affiliated with Apple or LiquidAI.
