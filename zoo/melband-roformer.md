# Mel-Band RoFormer (Kim Vocal) — Core AI

The zoo's **first source-separation** model — split any song into a **vocals (acapella)** stem and an
**instrumental (karaoke)** stem, entirely on device.
[`KimberleyJSN/melbandroformer`](https://huggingface.co/KimberleyJSN/melbandroformer) (MIT, ~228 M),
lucidrains' `MelBandRoformer` implementation, exported as **one Core AI graph** — STFT and iSTFT
included — with a Swift host that only frames and overlap-adds.

## Use it

Ships in the **[coreai-audio](../apps/coreai-audio)** app, **Separate** tab: load the demo clip or
pick a song from your library, get both stems back and play/export either. It pairs with the
**Music** tab (Stable Audio Open Small): generate a track, then rip its stems.

## How it works

Mel-Band RoFormer is `STFT → band-split (mel, overlapping) → axial rotary transformer ×6 → mask
estimator → band-average → complex mask multiply → iSTFT`. The neural core lowers to Core AI
directly — no rewrite, same as the Stable Audio DiT. Two moves make the on-device host trivial:

- **Real-arithmetic core** — the band-average scatter becomes a constant matmul and the complex mask
  multiply becomes real ops, so the graph carries no complex tensors and no `scatter_add` (neither
  lowers).
- **STFT/iSTFT folded into the graph as constant DFT matmuls** (window baked in). The shipped graph is
  `frames[1,2,801,2048] → recon[1,2,801,2048]`; the Swift host does reflect-pad, framing, and
  overlap-add — **no FFT, no vDSP packing**.

Fixed shapes throughout (8 s chunk = 352 800 samples, 801 STFT frames), so no
`--expect-frequent-reshapes` and no dynamic-shape recompiles. The instrumental stem is `mix − vocals`.

## Verification

| gate | result |
|---|---|
| re-authored real-arithmetic core vs the reference model | cos **1.0000000** |
| in-graph STFT/iSTFT (`SepFull2`) vs `torch.stft` reference | cos **0.9999984** |
| Core AI **fp16**, Mac GPU, framing + overlap-add round trip | cos **0.9999453** |
| **iPhone 17 Pro** (A19 Pro, AOT h18p) vs the Mac golden | cos **1.000000**, rms ratio **1.0000** |

On-device (iPhone 17 Pro, GPU): **8 s chunk in 1.23 s ≈ 6.5× real-time** warm (load 0.57 s); the
cold first run is 3.82 s ≈ 2.1× real-time (load 1.24 s) before the GPU clocks ramp.

Reproduce (`conversion/melband_roformer/README.md` has the full ladder):

```bash
PY=coreai-models/.venv/bin/python
$PY conversion/melband_roformer/export_full2_ship.py   # fp16 export + Mac GPU gate -> ship_macos/
xcrun coreai-build compile conversion/melband_roformer/ship_macos/mbr_full_fp16.aimodel \
  --output conversion/melband_roformer/artifacts/ios_full_h18p \
  --platform iOS --architecture h18p --preferred-compute gpu
bash conversion/melband_roformer/sideload_ios.sh <device-udid>   # then SEPARATE_SELFTEST=1
```

- 🤗 [MelBandRoformer-Vocal-CoreAI](https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI)
  — macOS `.aimodel` + iOS `.h18p.aimodelc` + self-test goldens.
- Base checkpoint: [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) (MIT).
- Attribution: Mel-Band RoFormer (Ju-Chiang Wang, Wei-Tsung Lu, Minz Won — ByteDance AI Labs);
  checkpoint by KimberleyJensen; lucidrains BS-RoFormer implementation; ZFTurbo training code.
