# Gemma-4 E2B mobile mixed-bit QAT weights — extraction + Core AI transplant (Phase 0 DONE)

**Goal:** match/beat LiteRT-LM's Gemma-4 iPhone decode (30.8 tok/s sustained, 0.146 J/tok,
4,074 tok/1% — the ONLY model where LiteRT beats Core AI on iPhone; see
`litertlm-convert/reports/litert-community-vs-mlx-coreai.md`) by transplanting Google's mobile
QAT mixed 2/4/8-bit weights into the Core AI gemma4 runtime. Core AI today: E2B VL 25.5 tok/s
(int4linsym, 2.0 GB). The gap driver is bytes/token, not kernels — LiteRT loses on every generic
model on Apple GPU (WebGPU delegate), it wins Gemma only via the recipe below.

## Phase 0 result (2026-07-02): recipe fully decoded, weights extractable

**Artifact:** `litert-community/gemma-4-E2B-it-litert-lm` / `gemma-4-E2B-it.litertlm` (2.59 GB)
→ `litertlm-convert/src_models/gemma-4-E2B-it-litert-lm/`.
**Extraction:** `litert_lm_builder.litertlm_peek.peek_litertlm_file(path, dump_dir, stream)`
(in `litertlm-convert/.venv`) dumps all 12 sections → `litertlm-convert/out/gemma4e2b_extract/`.
Per-tensor map (name/type/shape/bytes/nscale/nzp): `out/gemma4e2b_extract/decode_weight_map.json`
(parse = `ai_edge_litert.schema_py_generated`, mmap the .tflite, walk Model→Subgraph(0)→Tensors,
buffer size via `Buffers(t.Buffer()).DataLength()`).

### Sections (12)
| # | model_type | size | content |
|---|---|---|---|
| 0 | LlmMetadataProto | 12 KB | tokens, jinja template, gemma4 model_type |
| 1 | SP_Tokenizer | 4.7 MB | sentencepiece |
| 2 | tf_lite_embedder | 104 MB | input embed [262144,1536] **INT2** |
| 3 | tf_lite_per_layer_embedder | 1.28 GB | PLE table, **INT4** (1174 MB) — mmap'd, gather-only (~4.5 KB/tok read) |
| 4–6 | audio encoder/adapter/eoa | 104 MB | cpu-constrained |
| 7–9 | vision encoder/adapter/eov | 229 MB | fp16-activation |
| 10 | tf_lite_prefill_decode | 818 MB | **the transformer — mixed-bit map below** |
| 11 | **tf_lite_mtp_drafter** | 44 MB | INT4 36.7 + INT8 3.8 — **bundled spec-decode draft model** |

### The decode recipe (Section 10, 35 layers — per-channel SYMMETRIC linear: all zero_points are 0)
| weights | layers | bits | shape | notes |
|---|---|---|---|---|
| FFN gating1/gating2/down | L0–L14 | **INT4** | [6144,1536]×2 + [1536,6144] | these 15 layers also have own k/v (INT4, [256,1536] = 1 KV head × 256) |
| FFN gating1/gating2/down | L15–L34 | **INT2** | **[12288,1536]×2 + [1536,12288]** | **2× wider FFN at 2-bit = SAME bytes as 4bit@6144** (elastic capacity trade, bandwidth-neutral); KV shared from earlier layers |
| attn q / o | all 35 | **INT4** | [2048,1536] / [1536,2048] | 8 heads × head_dim 256 |
| PLE gate/proj | all 35 | **INT8** | [256,1536] / [1536,256] | + per_layer_model_projection [8960,1536] INT8 |
| embed ≡ lm_head (tied) | — | **INT2** | [262144,1536] | per-row scale, zp=0 |
| norms etc. | — | FLOAT32 | | 1.1 MB total |

- Quantization: `nscale = nzp = out_channels`, `qdim=0`, **zero_points stored but ALL ZERO ⇒
  dequant = code × scale[row]** (simpler than int4km — no LUT, no zp term).
  `bytes == nel*bits/8` exactly (dense packing, TensorType INT2/INT4 — the Python Interpreter
  API masks these as int8; parse the flatbuffer, not `get_tensor_details()`).
- **Packing convention VERIFIED** (vs MLX-4bit dequant oracle, embed rows mean-cos **0.8904**;
  wrong conventions give −0.34/≈0): INT2 = 4 codes/byte, first code in bits[1:0], signed two's
  complement (-2..1). INT4 assumed analogous (low nibble first, signed).
- Totals: INT2 383.8 MB / INT4 351.5 MB / INT8 41.3 MB ⇒ **active decode footprint ~777 MB**
  vs Core AI int4linsym E2B **2.0 GB** = **~2.6× fewer bytes/token** (decode is BW-bound; this
  is LiteRT's Gemma win, quantified).
- PLE tables: 35 × [262144,256] INT4 (1.17 GB, mmap/gather-only). Drafter: 22 tensors INT4/INT8.

### P1 DONE — portable extraction
`litertlm-convert/scripts/extract_gemma4_mixedbit.py` → `out/gemma4e2b_extract/
gemma4e2b_mixedbit_weights.safetensors` (335 tensors, 2.09 GB packed: embed 1 / ple_table 35 /
decode 277 / drafter 22) + `gemma4e2b_mixedbit_manifest.json` (per-tensor shape/bits/qdim/zp0).
Keys: `decode.layer_NN.{attn.{q,k,v,o}, mlp.{gating1,gating2,down}, ple.{gate,proj}}`,
`decode.lm_head`, `embed.composite`, `ple_table.compositeN`.

### MTP drafter (open question)
Runtime supports it (`kTfLiteMtpDrafter`, `litert_lm_engine_settings_set_enable_speculative_decoding`).
Swift `ExperimentalFlags.enableSpeculativeDecoding` defaults nil (= engine default); our bench app never
set it. **TODO: check litert-lm C++ engine default** — if spec-decode was ON in the 30.8 tok/s bench,
byte-parity alone should beat it; if OFF, 30.8 is pure bandwidth+kernel and we can ALSO add our own
spec-decode on top ([[project_spec_decode_port]]).

## Why transplant works (vs re-QAT)
These are Google's QAT weights FOR this exact config — transplanting the quantized values bit-exact
and dequantizing identically (per-channel affine) reproduces LiteRT's numerics by construction.
No training, no quality risk beyond runtime numerics. (The `q4_0-unquantized` HF release is a
DIFFERENT QAT run for uniform 4-bit — do not mix.)

## P2 progress (2026-07-02, same session)
- **All weight families verified vs MLX oracle** (`mlx-community/gemma-4-e2b-it-4bit` dequant),
  layer-0 cos: q .910 / k .944 / v .948 / o .915 / gate .926 / up .931 / down .943 / ple.gate .952
  / ple.proj .963 — cross-terms ≈0 ⇒ **gating1=gate_proj, gating2=up_proj** resolved.
- **Architecture = the public E2B arch exactly** (my initial "mobile differs" read was wrong):
  L4/9/14/19/24/29/34 = global-attn, head_dim **512** (q [4096,1536], kv [512,1536]); sliding
  layers head_dim 256 (q [2048,1536], kv [256,1536]); L15–34 FFN 12288 = `use_double_wide_mlp`.
  All of this is config-driven in `mlx_lm/models/gemma4_text.py` (global_head_dim / 
  num_kv_shared_layers / use_double_wide_mlp) → perfect reference implementation + oracle host.
- **fp32 norms extracted** → `gemma4e2b_fp32_norms.safetensors` (261 tensors,
  `scripts/extract_gemma4_norms.py`, consumer-scope attribution buffer-level across all 976
  subgraphs — names are anonymized `jax2tf_arg_N`/deduped `arith.constantN`):
  - 5 residual norms × 35 layers = real learned vectors (values ~0.05–75 ⇒ stored EFFECTIVE
    scale; HF/(1+w) convention ⇒ subtract 1 when mapping — verify in oracle).
  - **q_norm/k_norm = per-layer SCALARS** (nuniq=1, e.g. L00 q=0.9846 k=0.1269; L04 k=0.0648)
    = attention temperature per layer; k_norm absent L15+ by construction (no K computed).
  - **skip_scale = per-layer learned scalar** (`_maybe_apply_skip_scale`, e.g. L10=0.4438) —
    find its application point + MLX equivalent before the oracle run.
  - TODO: final_norm regex missed (the 1×(1536,) consumed by `StatefulPartitionedCall:N`) — grab it.
  - ple_projection_norm = real vector (256,).

## P2 DONE — transplant verified end-to-end (2026-07-02)
`scripts/gemma4_mixedbit_oracle.py` dequantizes everything and injects into
`mlx_lm.models.gemma4_text.Model` (540 params, `load_weights(strict=True)`), greedy-decodes.
Conventions settled by direct comparison with the MLX checkpoint (all cos ≥0.9987):
- norms = direct assignment (MLX `nn.RMSNorm` consumes effective scales — NO (1+w) shift);
- **`layer_scalar` exists in the public arch** (`h = h * layer_scalar` at layer end) and our
  extracted skip_scale matches it (L10: 0.4438 vs 0.443) — no module patch needed;
- q/k norms are constant vectors in the public checkpoint too (per-layer attention temperature);
- embed ≡ lm_head bit-identical (tied) — `tie_word_embeddings=True` correct;
- PLE = 35 × [262144,256] tables concat axis=1 → embed_tokens_per_layer [262144,8960].

**Greedy match vs `litert-mac-verify --greedy` (same prompt, "Why is the sky blue?"):**
14-token EXACT prefix ("The sky appears blue due to a phenomenon called **Rayleigh
scattering**. Here"), then a synonym fork ("Here's a breakdown" vs "Here is a detailed
breakdown") that MOVES with oracle dtype (bf16 "of the process" → fp16 "of why" = LiteRT's
wording) ⇒ near-tie fusion-numerics class, same PASS standard as prior device ports
(margin-fork precedent in [[project_gemma4_vl_port]]). **Transplant is numerically faithful.**

Bonus datapoint: LiteRT-LM Gemma4-E2B on Mac M4 Max (WebGPU): **decode 113.2 / prefill 52.2
tok/s** — their Mac decode BEATS our current E2B int4linsym engine (82.4 tbl) purely on bytes
(777MB vs 2.0GB), while their prefill is 20–80× slower than ours. Byte-parity thesis reconfirmed.

## Plan
1. ~~P1 extract~~ ✅  2. ~~P2 oracle~~ ✅
3. **P3 — Core AI receiving end**:
   - int8 per-channel: shipped. int4 per-channel affine: `gather_qmm_int4aff` machinery exists
     (MoE); need the dense form. **NEW: int2 per-channel affine dense matvec** (gemma4_metal
     family twin; 4-code variant of the int4km tg-mem staging — the fp4 kernel lesson applies:
     stage the dequant table/scales in threadgroup memory).
   - lm_head/embed INT2 matvec = the single biggest per-token read (100.7 MB table, per-row affine).
   - PLE INT4 gather: E4B port's PLE gather machinery exists at int8 → int4 variant.
4. **P4 — bench**: Mac llm-benchmark → iPhone PipelinedBench (A19) vs LiteRT 30.8 sustained
   (energy protocol from `litert-community-vs-mlx-coreai.md`). Success = ≥30.8 sustained at
   matched quality (greedy-match gate), with our prefill lead intact.

## Repro commands
```bash
cd ~/code/litertlm-convert
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('litert-community/gemma-4-E2B-it-litert-lm', local_dir='src_models/gemma-4-E2B-it-litert-lm')"
.venv/bin/python -c "
import sys
from litert_lm_builder import litertlm_peek as peek
peek.peek_litertlm_file('src_models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm','out/gemma4e2b_extract',sys.stdout)"
```
