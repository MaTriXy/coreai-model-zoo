# Video world models (V-JEPA 2) on Core AI — port notes

V-JEPA 2 (Meta) is a **self-supervised video encoder** — a "world model" that learns by predicting
in representation space (JEPA), not by generating pixels. Ported as
`facebook/vjepa2-vitl-fpc16-256-ssv2`: ViT-L backbone (3D RoPE attention over 16-frame clips) +
attentive pooler + Something-Something-v2 action head (174 physical-interaction classes). 375M,
fp16 ~675 MB — comfortably iPhone-sized.

## Shape of the port (ONE bundle)

Unlike the diffusion ports (3 bundles + host sampler), this is a single stateless graph:
`pixel_values_videos [1,16,3,256,256] → logits [1,174]`. Wrap `VJEPA2ForVideoClassification`
(`.logits`), direct-export with the usual externalize-drop {sdpa, rope}, run via `GraphModel`.
transformers ≥4.53 has the model class built in — no custom code.

## Lessons

- **One overlay: the no-op `squeeze` trap.** The reference `rotate_queries_or_keys` (3D RoPE) calls
  `emb_sin.squeeze(-1)` on a dim of size D/2 (=10). In torch, squeezing a non-1 dim is a NO-OP; the
  Core AI converter maps it to `ShrinkDims`, which REQUIRES size 1 → `Operation creation failed` at
  `VJEPA2RopeAttention`. Fix: monkeypatch the function with the squeeze removed (math unchanged) —
  see `conversion/vjepa2/export_fp16.py`. Generic rule: **a converter failure on a torch no-op is a
  patch-the-reference case, not a rewrite case.**
- Everything else direct-exports: engine vs reference **cos 0.999996, top-5 identical** (fp16 via
  `.half()` + fp16 reference inputs, as with Stable Audio).
- **Gate semantics, not just numbers.** The numeric oracle used random input; add a *semantic* gate:
  render a synthetic square moving up vs down (16 frames, ImageNet norm) and check the top labels
  flip (`Moving [something] up` ↔ `down`, with the optic-flow-equivalent camera-motion labels
  alongside — expected). Proves real video understanding on-engine in one script, no dataset needed
  (`conversion/vjepa2/gate_semantic.py`).
- **Slow-HF-day workflow:** `from_pretrained` can hang on network metadata even with a full cache.
  `HF_HUB_OFFLINE=1` fixes that but has a transformers bug (`checkpoint_files[0]=None`) when the
  snapshot is missing files — so **first finish the blob download with `hf_hub_download`** (it
  resumes `.incomplete` blobs), then run offline.
- Preprocessing lives on the HOST: 16 frames uniform-sampled, 256×256, RGB 0..1, ImageNet mean/std.
  (The model does NOT normalize internally.)
- **Perf (M4 Max GPU): ~150–180 ms per 16-frame clip, load 0.15 s** — real-time-ish video
  understanding; iOS AOT (h18p) compiles to ~1.3 GB.

## Why this model (EDGE)

Video SSL encoders are a vacuum on Apple silicon — no MLX port, no Apple stock path, edge deployments
target NVIDIA/Android. A camera-fed "what action is happening" demo is a zoo-first category
(video understanding), distinct from generation.
