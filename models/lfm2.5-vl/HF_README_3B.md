---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-VL-3B
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

# LFM2.5-VL-3B — Apple Core AI (`.aimodel`)

**LiquidAI's LFM2.5-VL-3B converted to Apple's Core AI** (the Core ML successor announced at
WWDC26), for macOS 27 and iOS 27. The detail tier of this family: where the
[450M](https://huggingface.co/mlboydaisuke/LFM2.5-VL-450M-CoreAI) answers *"two cats on a pink
couch"*, the 3B answers *"the cat on the left is smaller, with a gray and black striped coat,
while the cat on the right is larger with a brown and black striped pattern."*

Two bundles, run in sequence: a **SigLIP2-NaFlex vision tower + projector** (`patches
[1024,768] → image_embeds [256,2048]`, hidden 1152 × 27 layers) and the **LFM2 conv+attention
hybrid decoder** (hidden 2048, 30 layers = 22 short-conv + 8 GQA attention, vocab 128 000, tied
head), with the image tokens spliced in through a static `image_embeds` input. No recurrent
scan, so decode is loop-free on Apple's `coreai-pipelined` GPU engine with no custom kernels.

> Requires macOS 27 / iOS 27 (Core AI ships with the OS). Conversion code, gates and knowledge base:
> **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | measured (M4 Max) | numerics |
|---|---:|---|---|
| `gpu-pipelined/lfm2_5_vl_3b_vision_fp16` | 815 MB | **75.7 ms**/image | `image_embeds` cos **0.999995** vs fp32 HF |
| `gpu-pipelined/lfm2_5_vl_3b_decode_int8lin` | 3.1 GB | — | suite **7/9** cases token-exact; `logits_last` cos 0.999970 |
| `gpu-pipelined/lfm2_5_vl_3b_decode_int4lin` | **2.0 GB** | — | suite **7/9** — identical to int8 *and* to the fp16 baseline |
| `gpu-pipelined/lfm2_5_vl_3b_decode_int8lin_textcore` | 3.1 GB | **120.9 prompt / 105.3 decode tok/s** | the same weights with no image input |

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`. The tok/s row is the **text
core** because `llm-runner` cannot bind the VLM bundle's `image_embeds` buffer.

**int4 costs this model nothing**, which is worth stating plainly because the 450M sibling
craters at int4 (0 of 9 cases). Judged against an **fp16 baseline** rather than fp32 alone —
greedy decoding turns any near-tie into a different tail, and the fp16 bundle itself lands 7/9
— int8lin and int4lin both reproduce that 7/9. The divergences are wording: *"sleeping
peacefully on a bright pink couch"* → *"sleeping on a pink couch"*.

### iPhone 17 Pro — `ios-h18p/lfm2_5_vl_3b_decode_int4lin` + the fp16 tower

**27.5 prefill / 19.3–22.8 decode tok/s**, nat 16/16 and image oracle 24/24, clean at a
1024-token generation. int8lin does **not** load on iOS (its AOT `resources.bin` is 3.13 GiB);
int4lin's is 2.03 GiB and does — which is worth stating because the note this port was written
against put the load wall at 2 GiB, and 2.03 GiB was written up as expected-to-fail before a
phone was asked. It loaded. Use int8lin on a Mac and int4lin on a phone; on this model int4
costs nothing (7/9 on the suite, the same cases as fp16).

On device the description matches fp32's picture and diverges at the same near-tie the Mac
bundles take ("sleeping peacefully" → "sleeping"), then onto an equally accurate branch.

## Run it

```bash
git clone https://github.com/apple/coreai-models   # + the zoo's engine patches, see below
swift build -c release --product llm-runner

COREAI_CHUNK_THRESHOLD=1 .build/release/llm-runner \
  --model gpu-pipelined/lfm2_5_vl_3b_decode_int8lin_textcore \
  --prompt "The alphabet begins A, B, C," \
  --max-tokens 64 --sampling-strategy greedy \
  --inference-engine-variant coreai-pipelined --warmup off
```

The engine patches (`coreai-pipelined-extra-states` for the conv state,
`coreai-pipelined-static-inputs` for `image_embeds`) are in the zoo under `apps/`.

For the image path the host resizes to 512×512, normalizes `(x/255 − 0.5)/0.5`, and patchifies
into 16×16 patches with the **channel as the fastest axis** (`[y][x][c]`); then it runs the
vision bundle, binds the output as `image_embeds`, and rewrites the prompt's `<image>` ids
(124907) to `V + slot`. Reference implementation:
[`_smoke/lfm25vl_preprocess.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/_smoke/lfm25vl_preprocess.py).

**Two host details differ from the 450M and both are silent when wrong.** This checkpoint
declares `resample: 3` (PIL **BICUBIC**) where the 450M declares 2 (BILINEAR) — read it off
`processor_config.json`. And this tokenizer's post-processor does **not** prepend
`<|startoftext|>` (the 450M's does), while the chat template starts with it: feed the model a
prompt without BOS and it answers `" F, F, F, F"` — fluent degeneracy, no error.

## Converting this family yourself

Build the oracle on **transformers ≥ 5**: 4.57.6 applies the projector's LayerNorm
unconditionally while this config sets `projector_use_layernorm: false` and ships no such
weights, and `nn.LayerNorm`'s default init makes that invisible.

The weight shapes give away the rest: `patch_embedding.weight` is `[1152, 768]` — a **Linear
over pre-flattened patches**, not a Conv2d over an image — and `position_embedding.weight` is
`[256, 1152]`, a 16×16 grid **bilinearly resized (antialias) to the actual patch grid**. The
tower's 4304-wide MLP is not divisible by 32, so int8 there is per-block-**16**.

Everything is in
[`conversion/export_lfm25vl_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_lfm25vl_pipelined.py)
(`--hf-id LiquidAI/LFM2.5-VL-3B` — the same script that built the 450M) and
[`knowledge/lfm2.5-vl-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/lfm2.5-vl-port.md).

## License

LFM Open License v1.0, carried from
[`LiquidAI/LFM2.5-VL-3B`](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) (revision
`5a414ead75d45db003906d06fb62bd5b6846cec0`). Not affiliated with Apple or LiquidAI.
