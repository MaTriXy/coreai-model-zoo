# LFM2.5-VL-450M (vision-language) — Core AI

**The smallest VLM in this catalog by a factor of three.** A Core AI port of
[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M): image + text → text
on the [pipelined-engine fast path](../../knowledge/pipelined-engine.md), 658 MB for the pair.
Every other vision-language model here starts at ~2 GB resident, which is the difference between
a model an app *is* and a model an app *has*.

Architecture (`model_type: lfm2_vl`): a **SigLIP2-NaFlex tower** (hidden 768, 12 layers) whose
patch embedding is a **Linear over pre-flattened 16×16×3 patches** and whose 16×16 position table
is **bilinearly resized to the actual patch grid**; a 2-layer projector
(`pixel_unshuffle(2) → linear_1 → gelu → linear_2`, **no LayerNorm**); and the **LFM2 hybrid text
decoder already shipped in this repo** (hidden 1024, 16 layers = 10 short-conv + 6 GQA attention,
vocab 65 536, tied head, RoPE θ 1e6), reached by nothing more than a `model.language_model.` key
prefix. Only the tower and the projector were authored fresh —
[`knowledge/lfm2.5-vl-port.md`](../../knowledge/lfm2.5-vl-port.md).

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/LFM2.5-VL-450M-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-VL-450M-CoreAI)** —
`gpu-pipelined/lfm2_5_vl_450m_vision_fp16/` + `gpu-pipelined/lfm2_5_vl_450m_decode_int8lin/`
(the pair), `…_textcore/` (the same decoder with no image input), and `ios-h18p/` AOT variants of
all three, each gated on an iPhone 17 Pro. LFM Open License v1.0.

## What it is for

A 450M VLM answers scene-level questions — what is in the picture, where it is, what colours
dominate — and misses fine-grained geometry. That is the checkpoint, not the port: the same
weights on LiteRT/Pixel show the same split. Treat it as a caption/triage model that fits
beside an app, not as a document reader.

## Numbers (M4 Max, macOS 27.0 26A5378n, Xcode 27.0 27A5218g, coreai-torch 0.4.1)

| artifact | size | measured | numerics |
|---|---:|---|---|
| **vision fp16 (ship)** | **181 MB** | **18.0 ms**/image (median of 12, 512×512 → 256 tokens) | `image_embeds` cos **0.999996** vs fp32 HF |
| vision int8lin | 97 MB | 21.8 ms/image — **slower**, see below | cos 0.999729 |
| **decoder int8lin (ship)** | **477 MB** | text core: **609.2 prompt / 387.2 decode tok/s** | suite **7/9** cases token-exact; `coreai_gate.py` **PASS 16/16** |
| decoder fp16 | 717 MB | not benchmarked | logits cos **0.999994**; suite 8/9 |
| decoder int4lin | 349 MB | not benchmarked | suite **0/9** — **NO-GO**, see below |

`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`. The Mac tok/s row is the **text
core** (the same weights exported without the image input): `llm-runner` cannot bind the VLM
bundle's `image_embeds` buffer, so the text core is the Mac proxy — the same substitution the
[MiniCPM-V-4.6 card](../minicpm-v-4.6/README.md) documents.

## iPhone 17 Pro (AOT h18p, PipelinedBench, settled)

| bundle | prefill | decode | numerics |
|---|---:|---:|---|
| **`decode_int8lin` (the VLM bundle, image bound)** | **123.2** | **112.0** | nat 16/16 + **image oracle 24/24** |
| `decode_int8lin_textcore` | 122.1 | 110.6 | nat 16/16 + oracle 16/16 |
| `decode_int8lin`, `PB_G=1024` | 122.4 | 108.6 | nat 16/16, no collapse |
| **`vision_fp16`** | — | **33.6 ms**/image | **cos 0.999995** vs the Mac tower's own output |

The tower's first-ever encode costs **~860 ms** of on-device MPSGraph compile; every encode
after that is 33.6 ms (medians of 9 and 15 runs across two launches: 33.6 / 36.3). Warm it with
a dummy encode at load and the user's first photo is the warm number — the same lesson the
MiniCPM-V-4.6 port wrote down at ~2.7 s. Mac is 18.0 ms, so a phone pays 1.9x.

`xcrun coreai-build compile … --platform iOS --preferred-compute gpu --architecture h18p`
(**no** `--expect-frequent-reshapes`: on iOS it makes the runtime discard the AOT specialization
and compile on device, which SIGSEGVs with no log). Engine ready in 0.5 s warm / 4.2 s cold.

The VLM bundle and the text core measure the **same speed within noise** — binding a 256×1024
fp16 buffer costs nothing per step — which is what makes the Mac text-core proxy legitimate
rather than convenient.

The `PB_G=1024` row is mandatory, not extra: the iOS compiler miscompiles KV specializations at
seq ≥ 2048 and `g ≤ 256` cannot see it
([zoo PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6)). This bundle is clean at
1024.

**On device the image path is right but not token-identical to fp32.** The device describes the
same picture and drops one adjective at a near-tie: fp32 says *"two tabby cats … is stretched
out on its side with its p[aws]"*, the device *"two cats … is lying on its side with its head
resting on the"* — two forks, both word choice, with the ten tokens between them identical. The
gate therefore checks the **device-verified** sequence (recorded in PipelinedBench with the fp32
one beside it in `_smoke/lfm25vl_ref/vl_ref.json`), which is the same thing the MiniCPM-V-4.6
card does for the same class of fp16 near-tie.

The tower is gated on device against **its own Mac output**, not against fp32: what the phone
has to reproduce is the encode the decoder was gated with. cos 0.999995 says the h18p tower and
the h16c one compute the same image.

## Gates

The port is gated seam by seam against an fp32 transformers-5 oracle **before** anything was
exported (`_smoke/lfm25vl_ref.py` → `_smoke/test_lfm25vl_torch_ladder.py`), then again through
the engine:

- **fp32 torch ladder** — patch embeddings + resized positions, encoder layer 0, mid layer,
  post_layernorm, projector `image_features`: cos **1.000000** at every seam, and **48/48**
  token-exact greedy. Both at the image's native NaFlex grid (26×36) and at the 32×32 grid the
  bundle bakes.
- **host preprocessing** — the NumPy resize/patchify/normalize (`_smoke/lfm25vl_preprocess.py`)
  against the HF processor: patchify+normalize bit-exact, the antialiased resize within one
  0-255 level of Pillow, and **48/48** token-exact end to end through the model.
- **engine** — `_smoke/test_lfm25vl_aimodel_gate.py` chains the two bundles the way an app does
  (the vision bundle's own output feeds the decoder): vision cos 0.999996, decoder fp16 logits
  cos 0.999994, 48/48.
- **compression** — `_smoke/test_lfm25vl_suite_gate.py` over 9 image × prompt cases, and
  `conversion/coreai_gate.py` (arch `lfm2_5_vl`) on the text core, transcript beside this card.

Cosines are computed in float64; in float32 the reduction over ~10⁶ elements prints
`cos 1.000088` for two identical tensors.

## What compression costs, read rather than scored

int8 moves this decoder's logits much more than the family's larger members (`logits_last` cos
0.9866 vs 0.99992 for the 1.2B) — a 350M decoder has less redundancy per block. The fp16 bundle
at cos 0.999994 through the identical path is what proves that is compression, not a port bug.

Token counts against an **fp16 baseline**, not fp32 alone, because greedy decoding turns any
near-tie into a different tail: the fp16 bundle itself diverges on one of the nine cases. Both
int8 divergences are re-wordings ("contrast with the cats." → "contrast with the cats' fur.").

**int4lin is the cliff** — 0/9, and the failure mode is not broken repetition but fluent drift:
a kitchen becomes "a traditional *Italian* kitchen" where fp32 says "historical or rustic", and
one answer reports a kitchen's "Beige - visible in the *carpeted floor*". No loss curve catches
that; reading the generations does.

**The int8 vision tower is a Mac anti-optimization**: 97 MB instead of 181 MB, but cos 0.999729
instead of 0.999996, 21.8 ms instead of 18.0, and 6/9 suite cases instead of 7/9. The tower is
not bandwidth-bound at this size, so dequant is a net loss. It stays exported because a phone's
bandwidth budget is a different question — one no measurement here answers.

## How the image reaches the decoder

- The vision bundle is a plain `.aimodel`: `patches [1024, 768] → image_embeds [256, 1024]`,
  with the position-embedding resize baked as a constant for the fixed grid. The host
  patchifies (SigLIP2 NaFlex — the graph never sees an image).
- The decoder takes `image_embeds [256, 1024]` as a **static input** and the host rewrites the
  prompt's `<image>` ids to **extension ids `V + slot`**; in-graph
  `embed = ids < V ? embed_tokens[ids] : image_embeds[ids − V]`. Positions are plain 1D — no
  M-RoPE, no rope shift. This is the Qwen3-VL static-buffer recipe minus deepstack and minus the
  M-RoPE machinery.
- With no extension ids the decoder **is** a plain LFM2 text model — same bundle, no image
  required. The decoder carries three states (`keyCache`, `valueCache`, `convState`) and needs
  the pipelined extra-states patch plus `COREAI_CHUNK_THRESHOLD=1`.
- Driving decode from the Python runtime needs the **AOT** bundle
  (`--expect-frequent-reshapes --architecture h16c`); the raw `.aimodel` re-specializes per
  prompt length and thrashes.

## Convert / verify

```bash
# vision tower + VLM decoder (the published pair)
python conversion/export_lfm25vl_pipelined.py int8lin --vision-mode fp16

# the text-core proxy: same weights, no image input (benchmarks + coreai_gate)
python conversion/export_lfm25vl_pipelined.py int8lin --text-core --skip-vision

# oracles (transformers >= 5 ONLY — 4.x applies a projector LayerNorm this checkpoint does not have)
~/code/litertlm-convert/.venv-vl093/bin/python _smoke/lfm25vl_ref.py --resize 512x512
~/code/litertlm-convert/.venv-vl093/bin/python _smoke/lfm25vl_suite_ref.py

# gates
python _smoke/test_lfm25vl_torch_ladder.py --ref _smoke/lfm2_5_vl_450m_ref_512x512.npz
python _smoke/test_lfm25vl_aimodel_gate.py
python _smoke/test_lfm25vl_suite_gate.py --mode int8lin --show-text
python conversion/coreai_gate.py <text-core bundle> LiquidAI/LFM2.5-VL-450M -n 16
```

The 3B sibling shares this exporter (`--hf-id LiquidAI/LFM2.5-VL-3B`) but has **not** been
converted or gated here: its tower is wider (hidden 1152, 27 layers, MLP intermediate 4304 —
which forces int8 block 16, the same rule the MiniCPM-V SigLIP export hit) and its decoder is
30 layers over a 128k vocab. Nothing in this card transfers to it unmeasured.
