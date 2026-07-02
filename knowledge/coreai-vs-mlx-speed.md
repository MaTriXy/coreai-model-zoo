# Core AI vs MLX — where Core AI is faster, where it's slower, and *why*

A structured database of every measured Core AI–vs–MLX decode comparison we have, plus the
causal decomposition of the gap. All LLM rows are **same M4 Max, same protocol** as
`mlx-lm benchmark` (Apple's `llm-benchmark` is explicitly modeled on it): 512 prompt /
1024 generation / 5 trials, release build. MLX side = `mlx-lm 0.31.3`, `mlx-community` 4-bit.

## 1. The database (decode tok/s)

| # | Model | Arch class | Core AI | MLX | CA/MLX | Winner | Engine path | Dominant factor |
|---|---|---|---:|---:|---:|---|---|---|
| 1 | qwen3-0.6b | dense | **484** | 432 | 1.12 | **CA +12%** | pipelined | dispatch-bound, not BW-bound → MLX 4-bit edge doesn't cash in |
| 2 | qwen3-4b | dense | 145.4 | 145.8 | 1.00 | tie | pipelined | — |
| 3 | qwen3-8b | dense | **94.1** | 90.0 | 1.05 | **CA +5%** | pipelined | — |
| 4 | gemma3-4b-it | dense | **141.5** | 136.3 | 1.04 | **CA +4%** | pipelined | — |
| 5 | gemma3-12b-it | dense | 55.0 | 55.1 | 1.00 | tie | pipelined | biggest dense → BW starts to matter → MLX 4-bit pulls even |
| 6 | mistral-7b-v0.3 | dense | **101.7** | 97.5 | 1.04 | **CA +4%** | pipelined | — |
| 7 | gpt-oss-20b | **MoE** | 78.1 | **100.2** | 0.78 | **MLX +28%** | pipelined, stock GatherMM | **expert dispatch: GatherMM reads ALL experts/token (over-read-bound)** |
| 8 | Qwen3.6-35B-A3B | MoE (256e/top-8) | 30.9 | ~55–70 | ~0.5 | MLX | stock GatherMM | 32× expert over-read |
| 8b| Qwen3.6-35B-A3B | MoE + **gather_qmm kernel** | **64.9** | ~55–70 | ~1.0 | **tie/CA** | custom Metal sym8 gather | **kernel reads only routed experts → gap closes** |
| 9 | LFM2.5-8B-A1B | MoE (32e/top-4) | 39 → **141** | — | — | (3.6× self) | stock → gather_qmm | same over-read fix |
| 10| GLM-4.7-Flash | **MoE + MLA** | 20.3 → **52.4** | — | — | (2.6× self) | stock → gather_qmm | MoE fixed by kernel; **MLA on all 47 layers keeps it < qwen3.6** |
| 11| Qwen3-Coder-Next-80B-A3B | MoE (512e) | ~24 | "MLX-competitive" | ~1.0 | tie | gather_qmm | BW-bound on 79GB cold weight, not GDN |
| 12| Qwen3-ASR-1.7B (audio) | dense decoder, **ANE** | WhisperKit-ANE | **MLX 2.6×** | — | **MLX** | ANE (CoreML) | ANE = energy-not-speed; MLX-GPU wins raw tok/s + WER (1.52 vs 1.71) |

Sources: rows 1–7 `apple-models-bench.md` (head-to-head matrix); 8–11 `project_gather_qmm_kernel.md`,
`project_qwen36_moe_port.md`, `project_gather_qmm_next_target.md`; 12 `project_audio_understanding_qwen_omni.md`.

## 2. The one-line answer

**The difference is operator/architecture coverage on the engine — NOT the core engine.**
On standard **dense** transformers Core AI's pipelined engine ties or beats MLX. Core AI only
loses where the model uses an op-class the stock engine lowers *naively*:

- **MoE** → `SwitchGLU`/`GatherMM` lowers to a **dense matmul over ALL experts every token**
  (32× over-read for 256-expert top-8). MLX has real sparse expert gather. → MLX +28% on gpt-oss.
- **MLA attention** (GLM/DeepSeek) → naive materialized form does big per-token up-projections +
  full SDPA on every layer; the absorbed/latent-KV form that MLX-class runtimes exploit is hard
  to kernelize (the `c_kv` must be threadgroup-staged across heads, which our kernel didn't do).

When we replace the naive lowering with a **custom Metal kernel** (`gather_qmm`, reads only the
routed experts), the MoE gap closes: **Qwen3.6-35B-A3B 30.9 → 64.9 ≈ MLX**. So the gap was
dispatch/over-read, not the engine.

## 3. Factor decomposition (why a gap exists *at all* when it does)

The historical "MLX is ~2× faster, structural" verdict was measured on a **hand-rolled per-token
`fn.run()` loop** (~11% of BW peak, ~1000 Metal dispatches/token). That was the *loop's* ceiling,
not Core AI's. Apple's **`coreai-pipelined` engine** runs the same weights ~3.5× faster (qwen3.5
58.5 → 204 tok/s, ~2× MLX) with zero custom kernels — which is why the dense rows above tie/win.

The remaining gap, where it survives, decomposes into three independent multipliers:

| Factor | Size | Helps MLX when… | Notes |
|---|---|---|---|
| **Kernel coverage / dispatch** | ~2× | the model has uncovered op-classes (MoE gather, MLA) | dense is covered by the pipelined engine; MoE/MLA aren't → that's the whole gap |
| **Quantization byte-class** | ~1.5–2× | **bandwidth-bound** (big models, long ctx) | MLX = 4-bit affine g64; CA ships int8 (int4 flips argmax for non-QAT). Only pays off once BW-bound — see gemma3-12b tie vs 0.6b +12% win |
| **Host / framework / OS-runtime tax** | ~1.3× | always (uncontrollable) | the irreducible ~15–25% you don't own |

**Decision rule that predicts the winner:**
1. **Dense + pipelined engine** → Core AI ≥ MLX (Apple's tuned MPSGraph + async pipelined
   scheduling, and it isn't paying a Python-loop tax). The smaller / less BW-bound the model,
   the bigger Core AI's win (0.6b +12%); the bigger the model, the more MLX's 4-bit erases it (12b tie).
2. **MoE (sparse expert)** → Core AI loses on stock lowering (over-read), reaches **parity** with a
   custom gather kernel, but does **not** beat MLX — MLX's sparse dispatch is already good.
3. **MLA / exotic attention** → Core AI loses; the structural kernel (absorbed-MLA latent staging)
   is unsolved, so it stays below even Core AI's own dense models.
4. **ANE / iPhone** → not a raw-tok/s contest. MLX can win raw speed (Qwen3-ASR 2.6× over
   WhisperKit-ANE) but can't run on ANE/iPhone at all; Core AI's axis there is energy, always-on,
   and Foundation Models integration — not the tok/s race.

## 4. Takeaway for porting decisions

- Porting a **dense** model to Core AI: expect tie-or-win vs MLX for free on the pipelined engine.
- Porting a **MoE**: budget a `gather_qmm` custom Metal kernel up front, or you ship at ~0.5–0.78×
  MLX. With the kernel you reach parity (the ceiling), not a win.
- Porting **MLA**: parity/win is not currently reachable; ship for coverage/quality, not speed,
  until absorbed-MLA latent-staging lands.
- The genuine, non-speed moat is **ANE energy + iPhone reach + Foundation Models integration** —
  optimize for that axis, not for beating MLX on Mac-GPU tok/s.
