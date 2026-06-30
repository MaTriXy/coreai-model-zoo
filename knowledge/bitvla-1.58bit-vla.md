# BitVLA — 1.58-bit Vision-Language-Action on Core AI (the zoo's first VLA / robotics model)

Lessons from porting [`lxsy/bitvla-bf16`](https://huggingface.co/lxsy/bitvla-bf16) ([BitVLA,
arXiv:2506.07530](https://arxiv.org/abs/2506.07530), MIT) — a fully **1.58-bit ternary**
Vision-Language-Action policy (BitNet b1.58-2B LLM + BitSigLIP-SO400M vision) — to Apple Core AI,
running **image + instruction → 7-DoF robot action** on the iPhone 17 Pro GPU. Reuses the
[BitCPM ternary kernel](bitcpm-ternary-1.58bit.md).

## 1. What the checkpoint actually is

`bitvla-bf16` is a **bf16 latent master** (quantized on the fly), structured as LLaVA:
`vision_tower.vision_model.*` (SigLIP) + `multi_modal_projector.linear_1/2` (2-layer MLP, fp) +
`language_model.model.*` + `language_model.lm_head` (**untied**, vocab 128268). There is **no action
head and no proprio** in the base model — those live only in the LIBERO OFT fine-tunes. The base is
the **OXE-pretrained autoregressive** policy: it generates **7 discrete action tokens** from the
256-token tail of the vocab (README: "autoregressive next-action prediction following OpenVLA"). The
OFT path (`use_bi_attn=True`, bidirectional parallel decode + regression head) is for the LIBERO
fine-tunes — do **not** use it for the OXE base.

Quant formulas are identical to BitCPM/BitNet (`integrations/bitnet.py`, `modeling_siglip.py`):
**WeightQuant** = per-tensor absmean `round(W·s).clamp(-1,1)/s`, `s=1/mean(|W|)`; **ActQuant** =
per-token int8 `round(x·127/max|x|).clamp(-128,127)/(127/max|x|)`. Both LLM and the SigLIP linears
use W1.58-A8 (`vit_weight_bits 1 / vit_act_bits 8`).

## 2. Generalizing the ternary kernel

BitCPM's matvec assumed `K % 512 == 0` and `N % 32 == 0` (true for 4096/16384). BitVLA breaks both:
LLM down_proj K=6912 (%512=256), every SigLIP linear K∈{1152,4304}, fc1 N=4304 (%32=16).
`bitnet_ternary_metal.py` generalizes it:

- **Arbitrary K** (only K%16 for packing): the 32-lane × 16-code super-block still steps by 512, but
  a per-lane `k0 < K` guard zeroes the tail (K%16==0 ⇒ each lane's 16-code chunk is wholly in/out).
- **N padded** to a multiple of 32 (padded output rows computed then sliced).
- **Per-tensor (per-row) scale** for BitNet's absmean (vs BitCPM's per-256-block) — `D` is `[N,1]`.
- The `BitLinearMetal` wrapper applies **ActQuant before the kernel**, so it equals
  `F.linear(ActQuant(x), WeightQuant(W))` exactly. Construct from the bf16 master (codes =
  `round(W/mean_w).clamp(-1,1)+1`).

**Bug worth remembering:** the per-row scale buffer must be torch **`[N,1]`**, not `[1,N]`. The DSL
reverses axes, so the Metal `D[0, n]` reads `torch d[n,0]`. The torch reference (`d.reshape(-1,1)`)
is shape-agnostic so CPU passed either way — but the Metal kernel read out-of-bounds and produced
**NaN logits on the engine**. Same generalized kernel, M=1, validated engine-vs-torch cos 0.9997,
identical argmax.

## 3. VLM splice + action

- **inputs_embeds, not input_ids.** The LLM decode graph takes `inputs_embeds[1,1,2560]`; the host
  builds the sequence (text embeds + 256 projected vision embeds) and feeds it position-by-position
  (static-ids S=1 — the M=1 kernel is decode-only). Image token id (128010 at runtime via
  `set_constant`, not config's 128260) is moot when you splice by position.
- **Action-head slice.** The model only emits the 256 tail tokens, so slice the LM head to rows
  `[ACT_LO=128012 : 128268]` (256) → 656 MB → 1.3 MB, and decode argmax `j` → token `128012+j`.
- **Detokenize** (OpenVLA): 256-bin centers over [-1,1]; `bin = clip(total_vocab − token − 1, 0, 254)`;
  then BOUNDS-Q99 `0.5·(b+1)·(q99−q01)+q01` from the config `norm_stats` (27-dataset OXE mix; pick an
  `unnorm_key`, e.g. `bridge_orig`).
- Prompt: `System:…Human: <image>\nWhat action should the robot take to {instr}?<|eot_id|>Assistant: `,
  the `<image>` = 256 spliced embeds.

## 4. Validating against the official model (cheap oracle)

The official code (github ustcwhy/BitVLA) lives in a bundled **transformers fork** whose
`modeling_llava.py` hardwires `BitNetForCausalLM` + BitLinear-SigLIP. Run it in an isolated venv
(`python -m venv --system-site-packages` for the system torch, `pip install -e transformers --no-deps`,
`pip install "tokenizers>=0.21,<0.22"`), then `LlavaForConditionalGeneration(LlavaConfig(**config.json
minus norm_stats/n_action_bins/auto_map/architectures))` + `load_state_dict` = **0 missing / 0
unexpected**. Dump image embeds + action tokens as the oracle. Our standalone torch reference matched:
vision per-token **cos 0.9994**, full-pipeline action **6/7 tokens** with ~identical 7-DoF.

## 5. On-device gotchas (the part that took the longest)

- **The custom Metal kernel cannot JIT on device.** Loading the plain `.aimodel` low-level
  (`AIModel(contentsOf:)`) crashes the on-device compiler (`LLVM ERROR: cannot unwrap empty
  odiec_module_t`). It **must be AOT-compiled** (`xcrun coreai-build compile … --platform iOS
  --preferred-compute gpu --architecture h18p` → `.aimodelc`). (Standard-op graphs like the vision
  tower JIT fine.)
- **Loading the AOT `.aimodelc` low-level: `expectFrequentReshapes = false`.** The dynamic-shape LLM
  compiled with `--expect-frequent-reshapes` then loaded with `=true` fails `POSIX Code=2 "No such
  file or directory"`; with `=false` it loads. (The BitCPM precedent used the high-level engine,
  which takes input_ids; for inputs_embeds we go low-level: `AIModel`/`loadFunction`/`InferenceFunction.run`
  with mutable KV state via `MutableViews.insert(&keyCache, for:)`.)
- **Vision A8 act_quant stalls the iPhone GPU.** With the in-graph per-token round/amax activation
  quant, the first vision forward hung >10 min on h18p. Dropping vision to **fp16 activations**
  (ternary weights still baked; cos 0.997) runs in ~0.1–2.7 s. The rest of SigLIP (conv/LN/gelu/
  matmul) is GPU-fine.
- **Always `--architecture h18p`** on `coreai-build compile` — omitting it emits all ~20 Mac GPU
  archs (34 GB). Device install/launch need the screen **unlocked** (`CoreDeviceError 4016` = locked;
  set Auto-Lock = Never). Clear the captured `--console` log between runs (stale-log false alarms).

## 6. Why a ternary VLA on iPhone

VLA / robotics policies aren't in Apple's stock models or MLX; ternary VLA otherwise runs only on
bitnet.cpp (CPU). 1.58-bit shrinks a 7-DoF manipulation policy (vision + LLM) to ~2 GB resident on the
phone GPU — the durable Core AI edge (a kernel MLX lacks, on a device MLX doesn't ship to), for robotics.
