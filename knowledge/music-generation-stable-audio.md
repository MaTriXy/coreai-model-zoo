# Music / audio generation (Stable Audio) on Core AI — port notes

Text→music/audio is **latent diffusion**, same shape as LTX-Video / VoxCPM: a text encoder conditions
a DiT that denoises a latent over a few steps, then a VAE decodes the latent to a waveform. Ported
here as **Stable Audio Open Small** (`stabilityai/stable-audio-open-small`, 341M, Stability + Arm):
~11s of 44.1 kHz stereo from a prompt, fully on device.

## Shape of the port (3 bundles + a host sampler)

- **Conditioner** (`export_conditioner.py`): **T5-base** encoder (prompt→[1,64,768], pad-zeroed by the
  attention mask) + a **number conditioner** (seconds_total→[1,768], used as the +1 cross-attn token
  AND the global cond). Output `cross_attn_cond[1,65,768]` (=64 prompt + 1 seconds), `global_embed[1,768]`,
  `cond_mask[1,65]`. Tokenizer (T5 sentencepiece) runs on the HOST.
- **DiT** (`export_probe_dit.py`): the diffusion transformer (embed 1024, depth 16, qk-norm ln,
  rf_denoiser, cross-attn to prompt+seconds, prepended global cond). `x[1,64,256], t[1], cross_attn_cond,
  global_embed, cross_attn_cond_mask → v[1,64,256]`. Run 8×.
- **VAE** (`oobleck_vae.py` + `export_vae.py`): the Oobleck decoder `latent[1,64,256] → audio[1,2,524288]`.
- **Host sampler** (`capture_sampler.py`): 8-step rectified-flow euler, cfg 1.0.

## Lessons (the ones that saved days)

- **The DiT DIRECT-EXPORTS.** Wrap the *reference* `DiffusionTransformer` in a thin `nn.Module` (fixed
  inputs from a captured real call) and `export_to_coreai` with the externalize-drop of
  {scaled_dot_product_attention, rope} — no overlay rewrite. Engine output == reference, cos 1.0. Most
  of a diffusion model's risk is here; try the direct export FIRST.
- **The VAE decoder needs a CLEAN-REBUILD, not a wrap.** Wrapping the reference Oobleck decoder and
  exporting FAILS with `_apply(): Couldn't swap Conv1d.bias` / "has weakref" (the exporter's fp16
  `_apply`→`swap_tensors`). `remove_weight_norm`, `deepcopy`, fresh `nn.Parameter`,
  `set_swap_module_params_on_conversion(False)` ALL fail — the ref's weight_norm + alias-free snake +
  `model_half` carry weakrefs. **Fix (proven, VoxCPM/Kokoro): rebuild a plain-torch decoder with the
  SAME nn.Sequential nesting (so state_dict keys align) and FOLD weight_norm at load
  (`w = g·v/‖v‖_(dims≠0)`).** Bonus: the folded decoder is ~30× faster than the parametrized ref.
- **Derive the sampler by CAPTURE, not by reading.** Hook the DiT to record `(t, x_in, v_out)` per step;
  verify the update rule numerically. Here: 8 steps, `x_next = x + (t_next−t)·v` (rf-euler), cos 1.0 /
  mae 0 every step. Beats mis-reading a generic sampler dispatch.
- **fp16 bundles:** load the model `.half()` + pass fp16 reference inputs (export_to_coreai has no
  precision arg — precision follows the traced dtype). ⚠️the engine gate must pass **fp16** NDArrays to
  an fp16 bundle (fp32 → `CoreAIError 3`). In Swift, `fillNDArray(as: Float16.self)` (a `fill16` helper),
  not `Float`.
- **Run the 3 bundles via the kit's `GraphModel`** (CoreAIKitVision) with `TensorValue.float32(...)` —
  it auto-casts float32→the bundle's fp16 and reads back via `.floats()`. The app engine lives in the
  KIT (`CoreAIKit/StableAudio/StableAudioMusic.swift`) because Tokenizers + GraphModel both live there
  (adding raw swift-transformers to an app re-hits the device code-sign-bundle trap).
- **Perf (iPhone 17 Pro):** generate ~0.9s for 11.9s audio = **~13× real-time**, no cold-start penalty;
  one-time model LOAD ~3.9s (3 bundles + T5). A "4.8s" first-use = load + first generate; keep them
  separate in the UI.

## EDGE check BEFORE porting a music/audio-gen model (hard gate)

Music-gen moves fast on Apple Silicon — **check whether MLX / CoreML already ship it on iPhone**, or it's
a worse-MLX port (the DiffusionGemma / Qwen3-Coder-Next trap). Concretely:
- **Stable Audio OPEN SMALL (2025, 341M) = greenfield** when ported here — no turnkey on-device path → good.
- **Stable Audio 3.0 (medium/small, 2026) = DROP**: an MLX-Swift iPhone app already ships it
  (`kellyvv/StableAudio3-IOS`, same prompt→music UX) AND Stability provides an official CoreML path +
  MLX/TensorRT (`stable-audio-3-optimized`). Longer (up to 6:20) but no zoo edge. Uses a **T5Gemma**
  encoder (Gemma-licensed) + a SAME semantic-acoustic AE, ~2.5 GB, gated.
