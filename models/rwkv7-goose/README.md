# RWKV-7 "Goose" 1.5B — Apple Core AI port

**Community port — NOT an Apple model.** [`RWKV/RWKV7-Goose-World3-1.5B-HF`](https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF)
converted to Apple's **Core AI** runtime and validated on iPhone.

The first **pure-recurrent / linear-attention LLM on iPhone** with **O(1) per-token decode and NO
KV cache** — constant memory, unbounded context. Every layer is a WKV7 delta-rule matrix-state
time-mix + squared-ReLU channel-mix; the whole model carries just **two fixed-size states** and no
growing attention cache.

## Why this is interesting

Every transformer LLM on device grows a KV cache with context length — memory and per-token cost
rise as you generate. RWKV-7 is a **recurrence**: the entire history is summarised in a fixed
`[layers, heads, 64, 64]` matrix state (plus a small token-shift state), so **decode is O(1) in
memory and compute regardless of context length**. That is the durable edge of this architecture on
a phone.

- **On device (iPhone 17 Pro):** decode **25.2 tok/s**, O(1) 2-state, no KV cache.
- **Parity:** the exported Core AI graph is **teacher-forced top-1 127/127** vs a pure-torch
  reference (`max|Δlogits|` at fp32 = 7.6e-5); output is byte-identical to the reference.
- **No custom Metal kernel:** the WKV7 decode recurrence lowers to **standard Core AI ops** (like a
  Mamba-2 step), so it JIT/AOT-compiles cleanly.

## Contents

| Path | What |
|---|---|
| `aimodel/rwkv7_goose_1_5b/rwkv7_goose_1_5b.aimodel` | JIT Core AI decode model (run on macOS via `coreai.runtime`). |
| `h18p/rwkv7_goose_1_5b/rwkv7_goose_1_5b.h18p.aimodelc` | AOT-compiled for the iPhone GPU (`h18p`). |
| `h18p/rwkv7_goose_1_5b/rwkv_vocab.tsv` | RWKV World tokenizer, as `id\tbase64(bytes)` (build a byte trie, greedy longest match). |

**Quantization (`int8keepproj`):** the FFN and LM head are weight-only **int8** (per-block-32); the
recurrence projections (`r/k/v/o_proj`) and all LoRA factors stay **fp16** because the WKV7
delta-rule is precision-sensitive. Norms and embeddings stay fp16. This is the on-device ship recipe
(gated 127/127). ~2 GB.

## Architecture (config)

24 layers · hidden 2048 · head_dim 64 (32 heads) · FFN 8192 (sqrelu) · vocab 65536 (World
tokenizer) · untied head · pre-LN (norm_first) · WKV7 low-rank dims decay/a=96, v=64, gate=256.

**States (no KV cache):**
- `recState` `[24, 1, 32, 64, 64]` — the WKV7 matrix state `S` per head.
- `shiftState` `[24, 1, 2, 2048]` — token-shift previous hidden (slot 0 = time-mix, slot 1 = channel-mix).

**Per-token WKV7 recurrence** (per head, `S = [K, V] = [64, 64]`):
```
S = diag(exp w)·S − (kk·a)·(kkᵀ S);   S += k⊗v;   o = rᵀ S
```

## Usage (macOS, `coreai.runtime`)

The exported function `main` takes `input_ids [1,1]` + `position_ids [1,1]` (unused; RWKV-7 is
positionless) and the two states, and returns `logits [1,1,65536]`. States mutate in place across
S=1 steps. The chat format is `<eot(0)> + "User: {prompt}\n\nAssistant:"`; EOS is token 0.

```python
import numpy as np, coreai.runtime as rt
m = await rt.AIModel.load("aimodel/rwkv7_goose_1_5b/rwkv7_goose_1_5b.aimodel",
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
fn = m.load_function("main")
rec   = rt.NDArray(np.zeros((24,1,32,64,64), np.float16))
shift = rt.NDArray(np.zeros((24,1,2,2048),  np.float16))
res = await fn(inputs={"input_ids": rt.NDArray(np.array([[tok]], np.int32)),
                       "position_ids": rt.NDArray(np.array([[0]], np.int32))},
               state={"recState": rec, "shiftState": shift})   # rec/shift mutate in place
logits = np.asarray(res["logits"].numpy())[0, -1]
```

Reference decode + the parity gate + the World-tokenizer vocab builder live in the
[coreai-models-community](https://github.com/john-rocky) conversion scripts (`conversion/rwkv7/`).

## License

**Apache-2.0**, inherited from the source model
[`RWKV/RWKV7-Goose-World3-1.5B-HF`](https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF). This is
an independent community conversion for Apple Core AI and is not affiliated with or endorsed by
Apple or the RWKV authors.

---

**⬇️ Download:** [🤗 mlboydaisuke/RWKV7-Goose-1.5B-CoreAI](https://huggingface.co/mlboydaisuke/RWKV7-Goose-1.5B-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show rwkv7-goose-1.5b` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
