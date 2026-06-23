# Conversion guide (PyTorch → `.aimodel`)

Re-author the model with `coreai_models` primitives (so it lowers cleanly), export via
`torch.export`, convert with `TorchConverter`, optimize, save. Verify numerically against the
Hugging Face reference (cosine / top-1 argmax) before trusting a bundle.

## Canonical API (gotchas burned in)

```python
import torch, shutil
from pathlib import Path
from coreai_torch import TorchConverter, get_decomp_table
import coreai.runtime as rt

ep = torch.export.export(m.eval(), args=(), kwargs=inputs).run_decompositions(get_decomp_table())
prog = (TorchConverter()
        .add_exported_program(exported_program=ep, input_names=[...], output_names=[...])
        .to_coreai())
prog.optimize()
shutil.rmtree(out, ignore_errors=True)                 # save_asset will NOT overwrite
prog.save_asset(Path(out), rt.AIModelAssetMetadata())  # Path (not str); minimum_os defaults to v27

model = await rt.AIModel.load(Path(out), rt.SpecializationOptions.cpu_only())  # async; cpu_only for parity
fn = model.load_function("main")                       # SYNC; model.function_names is a list attr
res = await fn({"x": rt.NDArray(np_or_tensor)})        # __call__ is ASYNC -> dict[str, NDArray]
```

## Gotchas that cost real time

- **`save_asset` won't overwrite** — `shutil.rmtree(out)` first.
- **`AIModel.load` is async; `load_function` is sync; calling the function is async.** Mixing these
  up is the most common first error.
- **Use `cpu_only()` for numeric parity checks** — h16c (fp16-compute) on GPU/ANE produces larger
  hidden-state diffs on high-magnitude activations (still argmax-exact, but noisier maxdiff).
- **Unsupported ATen ops surface at `add_exported_program` validate time**, not at runtime. e.g.
  `aten.remainder.Scalar` (tensor modulo) is unsupported — compute it with `floor`/`where` or a
  scalar symint instead. Either register a lowering (`register_torch_lowering`) or avoid the op.
- **In-place state writes** (KV cache via `slice_update`) need `remove_functionalization(ep)` after
  `run_decompositions`, before converting. Without it the mutation is dropped.
- **Composite externalization** (`ExternalizeSpec` for RMSNorm/RoPE/SDPA/...) marks ops by *class*.
  If your export unit holds submodules of that class which are NOT in the traced graph (e.g. a
  front-end norm kept as an attribute), externalizing fails ("custom op not found"). Opt out by
  exposing `coreai_externalize_specs = ()` on the module, or pass an empty externalize list.
- **fp16 conversion**: keep RoPE `inv_freq` a plain fp32 attribute (NOT a buffer) so `.half()`
  can't underflow the small frequencies to zero. Cast RoPE cos/sin to the query dtype.
- **Truncating layers for smoke tests**: keep architectural regions (KV-shared / double-wide MLP)
  aligned with the checkpoint's real layout, or weights won't load.

## Verify, don't trust

Compare to an HF eager reference: cosine ≈ 1.0 and **top-1 argmax match** on a fixed prompt is the
real pass criterion. A large hidden-state maxdiff alone is usually h16c compute noise, not a bug —
let argmax/top-5 decide. Re-run parity after any OS/toolchain bump (the runtime is OS-coupled).

## Detection transformers

Found by the RF-DETR port (zoo's first detector, [zoo/rf-detr.md](../zoo/rf-detr.md)). Four
platform bugs, each pinned by a minimal repro; all four bite ANY model, not just detection —
detection transformers just happen to use the trigger ops (sine position embeds, bilinear
sampling masks, coordinate floors).

1. **`aten.arange` with float start/end/step aborts the converter** — C++
   `bad_optional_access`, no Python error. Repro: any graph containing `torch.arange(8.0)`
   (`torch.arange(8)` is fine; the *dtype* doesn't matter, the *argument types* do). DETR-class
   models hit it via `gen_sineembed_for_position(…, d_model / 2)` — a float dim. Fix: precompute
   the `dim_t` vector as a Python-list constant (kills the runtime arange/floordiv/pow chain too).
2. **int64-comparison bool chains clobber unrelated live buffers at runtime.** A chain like
   `((ix0 >= 0) & (ix0 < W)).to(float)` on `.long()` tensors makes a *different*, still-live fp
   tensor (two ops upstream; even a graph OUTPUT) read back garbage/NaN once the subgraph
   executes. Deterministic, unit-independent (CPU too); `clone()` / `contiguous()` barriers do
   NOT protect the victim; skipping `optimize()` doesn't either. Diagnosis pattern: a tensor is
   provably computed right (its other consumer is exact) but reads wrong later → buffer-liveness
   bug, hunt the comparison chain. Fix: compute 0/1 masks in float arithmetic —
   `1 - (x - x.clamp(lo, hi)).abs().clamp(max=1)` is exact on integer-valued floats.
3. **`aten.floor` / `trunc` / `ceil` lower to IDENTITY on the GPU delegate** (CPU correct;
   `round` rounds ties away-from-zero instead of to-even). Two natural workarounds also fail:
   `div(x, 1, rounding_mode="floor")` simplifies to identity, and `float→long→float` roundtrips
   are cast-cancelled by the converter **dropping truncation semantics (CPU too)**. The floor
   that survives every unit: `torch.div(x * 2.0, 2.0, rounding_mode="floor")` — floor-div with
   divisor ≠ 1 lowers correctly, and ×2/2 is a power-of-two scale (exact in fp). Corollary: the
   "compute remainder with floor" advice above must use THIS floor on GPU paths.
4. **`torch._assert` on data-dependent comparisons breaks torch.export non-strict**
   (GuardOnDataDependentSymNode, torch 2.11) — ironically added by upstreams *for* export
   compatibility. For static-shape exports the check is vacuous: no-op `torch._assert` around
   `torch.export.export` and restore after.

Detection-gate design note: gate detector numerics with **set-based matching** (per confident
oracle detection: same class, IoU ≥ 0.75, score within tolerance), not positional top-k
compare — DETR-family models emit near-duplicate predictions whose ranks swap under fp16/h16c
noise, and positional compare overflags healthy conversions. Random-noise inputs flip two-stage
top-k proposal selection at near-ties (the detection analog of the LLM argmax margin rule) —
use real images for the gate, noise only as an informational probe.

## Publishing & fetching artifacts (HF Xet)

Hugging Face now stores large LFS files in **Xet** content-addressed storage, and this bites both
download and upload of multi-GB `.aimodel` bundles:

- **`HF_HUB_DISABLE_XET=1` does NOT bypass Xet** — the `resolve/main/<file>` URL still 302-redirects
  to the Xet bridge (`cas-bridge.xethub.hf.co`), which reconstructs cold files server-side and can
  crawl or stall (the "stuck at a few MB" symptom).
- **`curl -C -` (resume) HANGS the Xet bridge** — the resume request sends a `Range:` header that the
  bridge stalls on (0 bytes forever). A **plain GET** of the same URL streams fine (measured 13 MB/s).
  If you must curl, no `-C -`; add `--retry`/`--speed-limit` to re-roll onto a warm edge.
- **Best for a flaky link: native `HF_XET_HIGH_PERFORMANCE=1 hf download <repo>`** — it fetches the
  reconstruction in parallel chunks and **resumes at chunk granularity** (from `~/.cache/huggingface/xet/`),
  so a dropped connection continues instead of restarting from 0 (which a single curl stream does).
- **Don't mix Xet and non-Xet attempts** for the same file: the cache snapshot symlink can end up
  pointing at a **sparse, incomplete** blob from the killed attempt (apparent size right, `du` actual
  size small) while the good bytes land in a differently-hashed blob. `from_pretrained` then fails to
  load it. Fix: repoint the symlink to the complete blob, or wipe `~/.cache/huggingface/hub/models--<org>--<name>`
  and re-download cleanly.
- **Upload:** `hf upload-large-folder <repo> . --repo-type=model` is **resumable** — kill it and re-run,
  it continues (only un-committed files re-upload). Uploads are fast when the chunks are already in the
  local Xet cache from a prior download.
