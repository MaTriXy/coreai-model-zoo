# Mel-Band RoFormer → Core AI (vocal / instrumental source separation)

Zoo's first **source-separation** model. Splits a song into a vocals (acapella) stem and an
instrumental (karaoke) stem, entirely on device.

- **Base**: [`KimberleyJSN/melbandroformer`](https://huggingface.co/KimberleyJSN/melbandroformer)
  (Kim Vocal), lucidrains `MelBandRoformer` impl, **License: MIT**. ~228 M params.
- **Runs**: Mac (fp16 `.aimodel`) + iPhone (AOT `.h18p.aimodelc`). Mac GPU ≈ 7.8× real-time.
- **App**: coreai-audio → **Separate** tab (pairs with the Music tab: generate a track, then rip its stems).

## How it's exported

Mel-Band RoFormer is `host STFT → [ band-split (mel, overlapping) → axial rotary transformer ×6 →
mask estimator → band-average → complex mask multiply ] → host iSTFT`. The `[...]` neural core lowers
to Core AI directly (no rewrite — same as the Stable Audio DiT). Two design moves make the on-device
host trivial:

1. **Real-arithmetic core** (`export_core.py::SepCore`): the band-average scatter becomes a constant
   matmul `A`, and the complex mask multiply becomes real ops, so the graph carries no complex tensors
   or `scatter_add` (which don't lower).
2. **STFT/iSTFT folded into the graph as constant DFT matmuls** (`export_core2.py::SepFull2`, window
   baked in). The final graph is `frames[1,2,801,2048] → recon[1,2,801,2048]`; the Swift host only
   does reflect-pad + framing + overlap-add — **no FFT, no vDSP packing**.

Fixed shapes throughout (8 s chunk = 352 800 samples, 801 STFT frames), so no `--expect-frequent-reshapes`.

## Reproduce

```bash
PY=../../../coreai-models/.venv/bin/python           # torch 2.9 + MPS
# deps: rotary_embedding_torch==0.3.5 ml_collections omegaconf beartype  (uv pip)
git clone https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model _ref/kim
HF_HUB_DISABLE_XET=1 hf download KimberleyJSN/melbandroformer MelBandRoformer.ckpt --local-dir _ckpt

$PY precheck_reference.py     # load (missing0/unexpected0) + separate a real song -> oracle
$PY gate_ladder.py            # SepCore vs reference:            cos 1.0000000
$PY export_probe.py           # SepCore export + Mac GPU gate:   fp32 cos 1.0
$PY gate_dsp_numpy.py         # numpy STFT/iSTFT == torch.stft:  cos 1.0  (Swift host recipe)
$PY export_core2.py           # SepFull2 (matmul STFT/iSTFT):    cos 0.9999984
$PY export_full2_ship.py      # fp16 export + Mac GPU gate:      cos 0.9999453  -> ship_macos/ + goldens
xcrun coreai-build compile artifacts/mbr_full_fp16/mbr_full_fp16.aimodel \
  --output artifacts/ios_full_h18p --platform iOS --architecture h18p --preferred-compute gpu
```

## Ship

- `ship_macos/` — `mbr_full_fp16.aimodel` (492 MB) + `metadata.json` + `golden_{raw,vocals}.f32`
  (demo clip + self-test target). Dev symlink → `apps/coreai-audio/Sources/SeparateAssets`.
- `ship_ios/` — `mbr_full_fp16.h18p.aimodelc` (AOT) + `metadata.json`. Push with `sideload_ios.sh <udid>`.
- Self-test: `SEPARATE_SELFTEST=1` (compares the 8 s golden chunk, cos + rms).

Attribution: Mel-Band RoFormer (Ju-Chiang Wang, Wei-Tsung Lu, Minz Won — ByteDance AI Labs);
checkpoint by KimberleyJensen; lucidrains BS-RoFormer impl; ZFTurbo training code. MIT.
