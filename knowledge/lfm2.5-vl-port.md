# LFM2.5-VL — port knowledge

LiquidAI's LFM2.5-VL (450M / 3B) is the **already-shipped LFM2 decoder with a SigLIP2-NaFlex
tower bolted on**. The decoder side is a key-prefix rename away from
[`lfm2.5-2.6b-port.md`](lfm2.5-2.6b-port.md); everything genuinely new is in the tower, the
projector, and the host preprocessing. This note is the part that is not the decoder.

Ported at 450M: `models/macos/lfm2_vl.py` (tower + projector + the decoder rider),
`conversion/export_lfm25vl_pipelined.py`, gates in `_smoke/test_lfm25vl_*`.

## The two shape tells

The weight shapes say what kind of SigLIP this is, and a port written from a MiniCPM-V or
Qwen-VL vision recipe gets both wrong while still producing fluent output:

```
[768, 768]   vision_tower...embeddings.patch_embedding.weight     Linear, NOT Conv2d
[256, 768]   vision_tower...embeddings.position_embedding.weight   a 16x16 grid, not the image's
```

1. **The patch embedding is a Linear over pre-flattened patches.** The graph never sees a
   picture — the host patchifies. That is SigLIP2 **NaFlex**: `pixel_values` arrives as
   `[n_patch, 16*16*3]`.
2. **The position table is 16x16 and gets bilinearly resized (antialias=True) to whatever patch
   grid the image produced**, per image. At a baked grid that resize is a load-time constant
   (`_init_positional_constants`); get it wrong and you lose cosine, not a crash.

The tower's own dims (hidden 1152 / intermediate 4304 on the 3B) happen to match MiniCPM-V's
SigLIP, which is exactly why the mismatch is easy to miss.

Projector: `pixel_unshuffle(2) -> linear_1 -> gelu -> linear_2`, **no LayerNorm**
(`projector_use_layernorm: false`, and no such weights ship). Two details worth spelling out:
the projector's gelu is the **exact erf** one while the tower's MLP is `gelu_pytorch_tanh`; and
HF's `pixel_unshuffle` names its dim 1 "width" and dim 2 "height", which reads backwards — the
tensor it is handed is `[1, patch_rows, patch_cols, d]`.

**Build the oracle on transformers >= 5.** 4.57.6 applies the projector LayerNorm
unconditionally; `nn.LayerNorm`'s default init (weight 1, bias 0) means a wrong reference with
no warning and no visible garbage. Full write-up in the 2.6B note, § "the same era, one layer
deeper". `_smoke/lfm25vl_ref.py` refuses to run on 4.x rather than let that reach a gate.

## Baking the grid, and what it costs

Tiling is input-dependent: for the 640x480 COCO fixture the processor emits
`pixel_values (1, 1024, 768)` + `spatial_shapes (26, 36)` and the projector returns
**234 tokens, not 256**. The zoo idiom is to bake one grid so the position resize and the
unshuffle fold to constants, and here the natural choice is exact: one 512x512 tile -> 32x32
patches -> 2x unshuffle -> **256 tokens = the checkpoint's own `max_image_tokens`**, with
`max_num_patches` (1024) fully packed, so no padding mask survives into the graph either.

The cost is the aspect ratio: a fixed square grid stretches a non-square image, which is the
one thing NaFlex exists to avoid. That is a real quality question this port does not answer —
it is not visible in any gate here, because the fixed-grid oracle is captured through the same
stretch.

Two oracles, deliberately: `lfm25vl_ref.py` at the image's native grid proves the tower math
(the padding mask is not needed — a masked key contributes nothing to an unpadded query, so the
authored module runs the unpadded prefix at grid (26,36) and matches row for row), and
`--resize 512x512` re-captures everything at the grid the bundle actually ships.

## Host preprocessing

`_smoke/lfm25vl_preprocess.py` is the spec, gated against the processor before any Swift.

- **Patch layout**: HF permutes `(b, C, ph, P, pw, P) -> (b, ph, pw, P, P, C)`, so inside one
  768-vector the **channel is the fastest axis** and the patch is row-major `[y][x][c]`. The
  natural `[c][y][x]` still produces a 768-vector, still runs, and still describes the wrong
  image.
- **Resize filter**: `resample: 2` is PIL BILINEAR, which on a downscale is an **antialiased
  triangle filter whose support grows with the reduction factor** — not the 2x2 tap that every
  GPU calls "bilinear". This is the part a CoreImage/vImage host has to match and the first
  thing to check if device output degrades. The NumPy implementation agrees with Pillow to
  within one 0-255 level (Pillow rounds its weights into 8-bit fixed point); fed through the
  model that gap moves `image_features` to cos 0.9998 and leaves the 48 greedy tokens
  unchanged.
- Rescale 1/255 then `(x-0.5)/0.5`; mean/std are 0.5 for all three channels.

## Gates

| gate | what it proves | result |
|---|---|---|
| `test_lfm25vl_torch_ladder.py` (fp32) | the re-authoring, seam by seam | cos **1.000000** at embeddings / layer 0 / mid / post_layernorm / image_features; **48/48** token-exact |
| the same at `--dtype fp16` | the ship dtype | logits cos 0.999976, 48/48 |
| the same with `--host-patches` | the NumPy host end to end | image_features cos 0.999801, **48/48** |
| `test_lfm25vl_aimodel_gate.py` | the exported bundles, chained (the engine's own image_embeds feed the decoder) | vision cos 0.999996; decoder fp16 logits cos 0.999994, 48/48 |
| `test_lfm25vl_suite_gate.py` | what the compressed bundle *says*, 9 image x prompt cases | see below |
| `coreai_gate.py` (arch `lfm2_5_vl`) | the text core through the release `llm-runner` | **PASS 16/16** |

Note the cosine numbers are computed in **float64**. In float32 the reduction over ~10^6
elements carries ~1e-4 of error — enough to print `cos 1.000088` for two identical tensors,
which makes every digit meaningless.

`coreai_gate.py` and `llm-benchmark` both drive the bundle through `llm-runner`, which has no
way to bind an `image_embeds` buffer, so both run the **text core** (`--text-core` exports the
same weights with no image input). The image path is gated by the suite gate instead.

## Quantization: measure by what it says, against an fp16 baseline

int8 on this decoder moves the logits far more than the family's larger members do —
`logits_last` cos **0.9866** through the engine, against 0.999994 for the same bundle in fp16
(the 1.2B's int8 sits at 0.99992). A 350M decoder has less redundancy per block; that is the
whole story, and it is not a port bug — the fp16 bundle proves the chain.

But a cosine on one position is not a quality measure, and neither is one prompt. The suite
gate runs 9 cases and **compares against an fp16 baseline, not against fp32 alone** — which
turns out to matter:

| decoder | size | cases token-exact | tokens |
|---|---:|---:|---:|
| fp16 | 717 MB | 8/9 | 406/432 |
| **int8lin (ship)** | **477 MB** | **7/9** | 410/432 |
| int4lin | 349 MB | **0/9** | 175/432 |

The fp16 bundle diverging on a case is the point: **greedy decoding turns any near-tie into a
whole different tail**, so without that baseline you would book it as compression damage. Read
the two int8 divergences and they are re-wordings — "contrast with the cats." vs "contrast with
the cats' fur.", "the walls and flooring" vs "the walls and floor".

int4 is the documented cliff, and it is worth seeing what the cliff looks like here, because it
is *not* broken repetition: every generation stays fluent and grammatical while the content
drifts — a kitchen becomes "a traditional **Italian** kitchen" where fp32 says "historical or
rustic", and one answer reports a kitchen's "Beige - visible in the **carpeted floor**". A
loss curve would not catch either. **int4lin: NO-GO for the 450M.**

**The int8 vision tower is a trap on Mac.** It is smaller (97 vs 181 MB) and worse on every
other axis: cos 0.999729 vs 0.999996, encode 21.8 vs **18.0 ms**, and 6/9 suite cases instead
of 7/9. At this size the tower is not bandwidth-bound on an M4 Max, so dequant overhead is a
net loss. It stays exported because a phone's bandwidth budget is a different question — but
that question needs a device measurement, not this one.

## What the 3B adds (same exporter, `--hf-id`)

The model code needed nothing: hidden 1152 / 27 tower layers and a 30-layer 128k-vocab decoder
come off the config, and int8 on the tower drops to per-block-16 by itself (4304 is not
divisible by 32). Everything below is what did NOT transfer.

- **The host resize filter is per-checkpoint.** The 450M declares `resample: 2` (PIL BILINEAR),
  the 3B `resample: 3` (BICUBIC). Read it; do not inherit it. And bicubic's negative lobes ring
  past both ends on hard edges, where **Pillow clips because it writes uint8** — a float
  implementation that does not clip agrees to cos 0.999997 and still differs by 15 levels at the
  ringing pixels.
- **BOS is not guaranteed by the tokenizer.** The chat template starts with `bos_token`, but
  whether `encode()` reproduces it depends on the post-processor: the 450M's is a `Sequence`
  that prepends it, the 3B's is a plain `ByteLevel` that does not. Prompt the 3B without
  `<|startoftext|>` and it answers `" F, F, F, F"` — fluent degeneracy, no error, and it looks
  exactly like a broken port. A host should add the BOS when the tokenizer did not.
- **int4 is not a family property.** The 450M craters (0/9, fluent drift); the 3B does not move
  (7/9 — the same cases as its own fp16 baseline). Same recipe, same suite, opposite verdict.
  The 2.6B text sibling behaved like the 3B. Read the generations of the model in front of you.
- **The 3B does not fit iOS.** Its int8lin AOT `resources.bin` measures 3.13 GiB and int4lin's
  2.03 GiB — measure the compiled artifact, not the `.aimodel`, since AOT expands it. Against the
  2 GiB (2^31) load wall that is a fail for int8 by a mile and for int4 by **30 MiB**, though the
  int4 case is inference from the earlier bracket (0.80 ✅ / 1.96 ✅ / 3.92 ❌) rather than a
  device run — worth actually trying, because 30 MiB is inside the noise of where that wall was
  ever pinned. The remaining lever is the 524 MB fp16 embedding, tied to the head and so not
  quantizable in place; past that it is a split graph.

Measured (M4 Max): vision 75.7 ms/image at cos 0.999995, text core 120.9 prefill / 105.3 decode
tok/s, decoder int8lin 3.1 GB / int4lin 2.0 GB / fp16 5.2 GB, tower fp16 815 MB.

## Runtime notes

- **Driving decode from the Python runtime needs an AOT bundle.** The raw `.aimodel`
  re-specializes per prompt length and thrashes (tens of GB of scratch, no output):
  `xcrun coreai-build compile <name>.aimodel --platform macOS --preferred-compute gpu
  --expect-frequent-reshapes --architecture h16c`, then load the `.aimodelc` with
  `SpecializationOptions.default()`. Same lesson as the LFM2-Audio backbone.
- The decoder carries **three states** (`keyCache`, `valueCache`, `convState`) and needs the
  pipelined extra-states patch, plus `COREAI_CHUNK_THRESHOLD=1` — prefill runs as S=1 steps.
- `image_embeds` is an extra graph input. The engine classifies extra inputs as *static* when
  the caller supplies a buffer for them and as *per-token* otherwise, so a caller that supplies
  nothing fails with "declares per-token input(s) image_embeds but perTokenInputProvider is
  nil". That is the whole reason the text-core proxy exists.
- Not a thinking model: the generation prompt does not open `<think>` (unlike the 2.6B).
  `tokenizer_class: "TokenizersBackend"` + a separate `chat_template.jinja` apply here too —
  `save_tokenizer` in `export_lfm2_decode_pipelined.py` handles both.

## Measurements

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`.
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`, on the int8lin **text core**:
**609.2 prompt / 387.2 decode tok/s**. Vision encode: 18.0 ms/image (fp16, median of 12).

Ship pair = vision fp16 (181 MB) + decoder int8lin (477 MB) = **658 MB**.

iPhone 17 Pro, AOT h18p, PipelinedBench, settled: the VLM bundle **123.2 prefill / 112.0 decode
tok/s** with the image buffer bound (nat 16/16 + image oracle 24/24), the text core 122.1 /
110.6. **Binding a 256x1024 fp16 buffer costs nothing per step** — the two differ within noise,
which is what makes the Mac text-core proxy legitimate rather than merely convenient. A
`PB_G=1024` run is mandatory for any dynamic-KV bundle (the iOS compiler miscompiles KV
specializations at seq >= 2048 and g <= 256 cannot see it); this one is clean at 122.4 / 108.6.

On device the image path is **right but not token-identical to fp32**: the same picture, one
adjective dropped at a near-tie ("two tabby cats … stretched out on its side with its p[aws]" ->
"two cats … lying on its side with its head resting on the"), with the ten tokens between the
two forks identical. Record the DEVICE sequence as the gate and keep the fp32 one beside it —
the same thing the MiniCPM-V-4.6 port did for the same class of fp16 near-tie. A device gate
that demands fp32-exactness on a VLM will fail on wording and teach you nothing.

The tower on device: **33.6 ms/image** (median of 9; 36.3 on a second launch's median of 15)
against 18.0 on the M4 Max, and **cos 0.999995 vs its own Mac output** — gate a tower against
the encode the decoder was gated with, not against fp32, or you are measuring two things at
once. The **first-ever** encode costs ~860 ms of on-device MPSGraph compile, which a dummy
encode at load moves off the user's first photo (MiniCPM-V-4.6 saw ~2.7 s for the same reason).
`PB_VISION=<dir>` in PipelinedBench runs this: it shapes everything from the graph's own
descriptors, so it fits any single-input/single-output tower.
