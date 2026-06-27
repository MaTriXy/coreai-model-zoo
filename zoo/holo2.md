# Holo2-4B — Core AI

[🤗 mlboydaisuke/Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) · Apache-2.0 · base [Hcompany/Holo2-4B](https://huggingface.co/Hcompany/Holo2-4B)

H Company's **computer-use / GUI-grounding** VLM: given a screenshot + an instruction
("click the submit button") it predicts the **click coordinates / locates the UI element**
(SOTA UI localization). Built on the **Qwen3-VL-4B** backbone, converted to Apple **Core AI**.
The zoo's **first GUI-grounding / computer-use model**, and a worked example of riding an existing
zoo pipeline: Holo2-4B is byte-identical to Qwen3-VL-4B, so the conversion is the stock
`export_qwen3_vl_pipelined.py` with `--hf-id Hcompany/Holo2-4B` — no model-code changes.

## Parity (vs fp32 HF oracle, Core AI GPU engine)

| stage | metric |
|---|---|
| **vision** (`holo2_4b_vision`) | image-embeds cos **0.999983**, deepstack cos **0.999989** — PASS |
| **decoder** (`holo2_4b_decode_int8lin_s1`) | S=1 sweep **4/4**, **16/16** decode steps token-exact, HF-seeded match — PASS |

## Contents
- `gpu-pipelined/holo2_4b_decode_int8lin_s1/` — decode bundle (static query=1, per-block-32 int8
  linear body). Rides Apple's `coreai-pipelined` GPU engine and **specializes on-device — no AOT**
  needed (the static decode graph is cheap to specialize, unlike a dense 4B *dynamic* bundle).
- `gpu-pipelined/holo2_4b_vision/` — fixed-grid vision encoder `.aimodel` (fp16): `patches
  [784,1536] -> (image_embeds [196,2560], deepstack [3,196,2560])`. Run once per image.

## Conversion

- **Stock Qwen3-VL pipeline.** `coreai-models/.venv/bin/python conversion/export_qwen3_vl_pipelined.py
  int8lin --hf-id Hcompany/Holo2-4B` → decoder (+ `_s1` gate twin) + vision. text hidden 2560 /
  36 layers / 8 KV / head_dim 128 / vocab 151936; vision `qwen3_vl` depth 24.
- **Why Holo2 (not Holo1.5 / Holo-3.1):** Holo evolved 1.5 (Qwen2.5-VL) → 2 (Qwen3-VL) → 3.1
  (Qwen3.5-VL). Holo2 is the generation whose backbone matches the zoo's shipped Qwen3-VL pipeline,
  so it drops in with an HF-id swap; Holo-3.1 would need a new `qwen3_5_vision` tower.
- int8lin body, fp16 vision. (int8hu — untied absmax int8 head — is the optional head-quality upgrade,
  same as the other pipelined riders.)

## Run

In the zoo's **CoreAIChat** app: pick **Holo2 4B**, attach a screenshot, and ask where an element
is / what to click — it grounds the instruction to the image and returns the location. Rides the
same on-device path as [`qwen3.5`](qwen3.5.md)'s VLM siblings (Qwen3-VL decoder + vision tower).
