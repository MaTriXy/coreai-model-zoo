# Gemma 4 E2B vision (VL) — Core AI

**The zoo's second VLM, and the first multimodal Gemma on Core AI**: image +
text → text on the GPU [pipelined-engine fast path](../../knowledge/pipelined-engine.md),
riding the **already-shipped gemma4 text decoder** — same checkpoint, same PLE
tables, no engine changes beyond the published static-inputs patch.

Architecture (`google/gemma-4-E2B-it`, ship pairing =
`google/gemma-4-E2B-it-qat-q4_0-unquantized` which contains the FULL
multimodal model): the existing **gemma4 text decoder** (35 L, dual
sliding/full attention, KV-sharing, PLE) + a **SigLIP-class ViT vision tower**
(16 L, h768, MHA-12, head_dim 64, patch 16 as a flat Linear, 2D rope θ=100,
QK-norm + scale-free V-norm, **checkpoint-calibrated activation clamps on
every linear** — 224 active clip sites) + a 3×3 average pool and a 768→1536
projection (`embed_vision`).

**No M-RoPE, no DeepStack** — and the decisive simplification:

> **The image span is CAUSAL on E2B/E4B.** `use_bidirectional_attention` is
> `None` in their text_config — HF's modeling routes small Gemma 4 models
> through a conventional causal mask (only the 26b/31b set `"vision"` and
> attend bidirectionally within an image). Verified two ways: the mask
> construction in `modeling_gemma4.py`, and a runtime dump of the materialized
> fp32 masks (image span [5,260]: **0/32640 cells allowed above the
> diagonal**, both full and sliding). A causal-KV engine runs it as-is.

Soft tokens are **variable by aspect ratio** (= patches/9): a square image
resizes to 768×768 = 48×48 patches = **256 tokens**; the 280 budget needs
10:7. The fixed-grid vision encoder bakes the square grid; the decoder's
`image_embeds` static input is sized at the 280 max and the host fills
`n_soft` rows.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI)** —
`gpu-pipelined/gemma4_e2b_qat_vl_decode_int4linsym_tbl/` (Mac decoder),
`gpu-pipelined/gemma4_e2b_qat_vl_decode_int4linsym{,_aotc_h18p}/` (iPhone
decoder, provider mode + the h18p AOT) and
`gpu-pipelined/gemma4_e2b_qat_vl_vision/` (fixed-grid vision encoder, fp16,
296 MB). Pair them with the QAT PLE tables already on the repo
(`ios-frontend/gemma4_qat_gather_raw`) — bundle + tables + vision are a
checkpoint-derived triple (the QAT run retrains `embed_vision`, so the vision
encoder must also come from the QAT checkpoint).

## How the splice works (3 lines on top of the text graph)

The host rewrites the prompt's 256 `<image_soft_token>` ids to **extension
ids** `V + slot` and binds `image_embeds [280, 1536]` (vision output, raw —
no √h embed scale) as a static input. In-graph:

- `x = ids < V ? embed_tokens[ids] : image_embeds[ids - V]`
- the PLE gather indexes `where(ids ≥ V, pad_token_id, ids)` — HF computes
  per-layer rows for the **PAD token (id 0)** at image positions
  (`llm_input_ids[mm_mask] = pad`), so the PLE tables stay byte-identical to
  the text ship; the per-layer *projection* branch reads the spliced `x`,
  exactly like HF.
- positions/rope/masks/KV: **unchanged** (standard contiguous positions).

Text-only prompts degenerate to the text decoder (one dead 1-row gather).

## Numbers

| platform | decoder config | prefill | decode | notes |
|---|---|---|---|---|
| M4 Max | int4linsym **tbl** | **95.2** | **82.4** tok/s | ≥ the text tbl (87.1/77.0) — the splice costs nothing |
| iPhone 17 Pro | int4linsym **provider** (aotc h18p) | **41.2** | **25.5** tok/s | settled; footprint 1.96 GB, headroom 4.5 GB |

Vision encode: ~100–170 ms per image (fp16, one shot). Engine ≡ python
**24/24** on the Mac (both tbl and provider modes, token-for-token through a
knife-edge tie). On-device the 64-token rollout describes the test scene
completely and correctly (building, roof, sky, ground, doorway, sun); the
chain forks from the Mac sequence at one **deterministic margin-1.08 synonym**
("stylized"→"minimalist") — AOT-fusion fp16 class at a 272-token context,
the same envelope the Mac runtime itself shows at the prompt tail.

## Two findings that outlive this port

1. **QAT q4_0 checkpoints need plain absmax (`--lin-sym`) at long horizons.**
   The default `symmetric_with_clipping` int4 grid mismatches the q4_0 grid
   the QAT weights were *trained* for; the error compounds with context — at
   272 tokens it flips real-margin argmaxes (oracle top-2 gaps of 1.0+) and
   drags prompt-end logits cos to 0.75. A 19+8-position gate cannot see this
   ("clipping ≈ sym" at short horizons is an artifact). Gate VLMs — and
   re-gate QAT text bundles — at real context lengths.
2. **iOS scratch-heap ceiling (engine bug, second reproducer).** The tbl
   variant's in-graph table gather + the VL splice overflow MPSGraph's
   ~208 KB per-encode scratch heap on iOS at the FIRST encode
   (`allocateMTLBufferFromMTLHeap … exceeds heap total 212992` + ViewOp
   abort) — the same class the 12B dense port hit with 16-head full
   attention. Model-side graph squeezing is blind (a 36 KB-smaller dequant
   chain crashed at the byte-identical offset). macOS is unaffected; the
   device ships **provider mode** (per-token PLE rows via
   `PerTokenInputProvider`, `image_embeds` still a static buffer — the
   patch routes static and per-token extra inputs simultaneously).

Gating used the **fp32-oracle margin rule**: an argmax flip where the
oracle's top-2 logit gap is < 0.1 is a knife-edge tie (fp16 class), not a
failure — this prompt has exactly two (margins 0.056 / 0.028), and every
real-margin position must match exactly.

## Convert it yourself

```bash
# decoder (Mac tbl + iPhone provider) + fixed-grid vision encoder
python export_gemma4_vl_pipelined.py int4lin --lin-sym --tbl \
    --raw-dir …/gemma4_qat_gather_raw     # Mac ship
python export_gemma4_vl_pipelined.py int4lin --lin-sym --skip-vision  # device ship
```

Model overlay: `models/macos/gemma4_vision.py` (fixed-grid tower +
embed_vision; activation clamps become plain-bool-gated `torch.clamp`s at
load — a tensor-valued `isfinite()` check is a data-dependent guard
torch.export rejects) and the two `Gemma4VLPipelined*ForCausalLM` subclasses
in `models/macos/gemma4_pipelined.py`. Run with
`COREAI_CHUNK_THRESHOLD=1` + the static-inputs patch (tbl also binds
`ple_table`/`ple_scale`; provider sets the per-token provider and maps
extension ids → the pad row host-side).
