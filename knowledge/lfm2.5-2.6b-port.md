# LFM2.5-2.6B — port knowledge

`LiquidAI/LFM2.5-2.6B` (LFM Open License v1.0) is the second LFM2 checkpoint in the zoo, after
[LFM2.5-1.2B](../models/lfm2.5/README.md). Architecturally it is the *same model, scaled*, so the
interesting content here is not the network — it is **what a transformers-v5-era checkpoint does to
an exporter written against transformers 4.x**, which is the part that generalizes to every
LiquidAI release from mid-2026 on, and to the LFM2.5-VL towers next.

## The port was a config change, not a port

`export_lfm2_decode_pipelined.py --hf-id LiquidAI/LFM2.5-2.6B` is the whole conversion. The
authoring module reads `config.json` generically, so scaling came for free:

| | 1.2B | 2.6B |
|---|---|---|
| layers | 16 = 10 conv + 6 attn | **30 = 22 conv + 8 attn** |
| hidden / MLP | 2048 / 8192 (auto-adjusted from 12288) | 2048 / **10752** (`block_auto_adjust_ff_dim: false`) |
| heads | 32 q / 8 kv, head_dim 64 | same |
| vocab | 65 536 | **128 000** |
| RoPE θ | 1e6 | **1e7** |
| conv state | `[10, 1, 2048, 2]` | `[22, 1, 2048, 2]` |

Still no recurrent scan — the conv mixer is a 3-tap depthwise causal conv — so the decode graph
stays loop-free and the single fixed-shape conv state stays inside the
`coreai-pipelined-extra-states` budget. No engine change.

Note `block_auto_adjust_ff_dim` flips to `false` at 2.6B. The 1.2B derives its MLP width
(`2/3 · 12288`, rounded to a multiple of 256 → 8192); the 2.6B states 10752 directly. Code that
hardcodes the 1.2B's derivation silently builds the wrong width.

## Two silent config traps (the transferable part)

Neither raises. Both produce a bundle that exports, loads, runs, and is wrong or unusable.

### 1. RoPE theta moved into `rope_parameters`

```json
// 1.2B                          // 2.6B  (transformers-v5 era)
"rope_theta": 1000000.0          "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}
```

`lfm2_config_from_dict` read `raw.get("rope_theta", 1e6)`. On the 2.6B that returns the *default* —
a 10× wrong θ, every position mis-rotated, no error anywhere. The oracle gate would have caught it,
but only if you ran the gate; an exporter that trusts a clean run does not.

`tie_word_embeddings` has the same shape of problem: the authoring name is `tie_embedding`, the HF
name is `tie_word_embeddings`, and the default is `True`, so an untied checkpoint would silently be
built tied. Both keys are now read under either name.

**Rule: on a new checkpoint of a family you already support, diff the config keys against the one
you ported, not just the values.** A moved key reads as a missing key, and a missing key reads as
the default.

### 2. `tokenizer_class: "TokenizersBackend"` does not exist in transformers 4.x

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

Raised by `AutoTokenizer.from_pretrained` — **after** the multi-GB `.aimodel` has been written, so
the failure leaves a bundle that is one directory short of loadable and looks like a completed
export in every way except the missing `tokenizer/`.

The tokenizer itself is fine: it is an ordinary `tokenizer.json` next to that config. Two things to
get right when loading it directly:

- carry `bos_token` / `eos_token` / `pad_token` / `model_max_length` over from `tokenizer_config.json`
  into `PreTrainedTokenizerFast`, and
- copy **`chat_template.jinja`**. This checkpoint generation keeps the template in its own file
  rather than inside `tokenizer_config.json`, so the obvious fix ships a bundle with no chat
  template at all — which `zoo_verify.py` fails a published repo for, and which turns a chat model
  into a raw completer for anyone who does not notice.

`export_lfm2_decode_pipelined.py:save_tokenizer` does both. Gate the result: encode/decode
round-trip against the source `tokenizers` library, including CJK and emoji, and check the special
token ids. Ours passed 6/6 with `eos <|im_end|> 124900`.

### The same era, one layer deeper: transformers 4.x can also be *wrong* rather than absent

Building the LFM2.5-VL oracle in the same 4.57.6 environment turned up the nastier version of this.
`Lfm2VlMultiModalProjector.forward` in 4.57.6 applies its LayerNorm unconditionally:

```python
# 4.57.6                                   # 5.14.1
self.layer_norm = nn.LayerNorm(in_ch)      self.layer_norm = nn.LayerNorm(in_ch) if config.projector_use_layernorm else None
...                                        ...
x = self.layer_norm(x)                     if self.use_layer_norm:
                                               x = self.layer_norm(x)
```

Both VL configs set `projector_use_layernorm: false` and ship no such weights, so 4.x creates them
fresh. `nn.LayerNorm`'s default init is weight 1 / bias 0 — which is an identity *affine* but still
normalizes — so there is no garbage output to notice, only a quietly different reference. **An
oracle built there would have certified a wrong port as PASS.** Run VL oracles on transformers ≥ 5;
`_smoke/lfm25vl_ref.py` refuses to run otherwise and says why.

## Quantization

`int8hu --head-sym` is the ship, for the same reason as the 1.2B but larger: the head is
128 000 × 2048 = 262 M parameters, 524 MB of fp16 reads per token. Untying and quantizing it to int8
buys **+8.6 % decode** (1.2B: +9.3 %) and *grows* the bundle 3.2 → 3.4 GB, because untying stores an
fp16 embedding plus an int8 head instead of one shared table. `int8lin` is therefore not worth
publishing for this model — slower and only 0.2 GB smaller.

**`int4lin` did not hit the cliff**, which is the surprise. The family's int4 history is bad —
`lfm2.5-8b-a1b-moe`, gemma4-12B and the Qwen attempts all flip a high-margin token and degrade into
broken repetition. Here the 16/16 gate passed and four long greedy generations (algorithmic
explanation, iterative Fibonacci with complexity analysis, clock arithmetic, a Japanese
instruction) read clean end to end: 2.0 GB at 139.2 tok/s. Four prompts are not a benchmark —
record this as *no cliff observed on a dense 2.6B*, and keep reading generations on the next one.

Of that 2.0 GB, **524 MB is the still-fp16 embedding** (embeddings, conv1d, norms and the attention
q/k/v/out projections stay high precision in every quantized mode — the attention projections
because quantizing them flips near-tie argmaxes, 14/16 instead of 16/16). If this model is ever
sized for a phone, the embedding is the remaining lever, not the layers.

## Measurements

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`,
`llm-benchmark -p 128 -g 256 -n 3`, `COREAI_CHUNK_THRESHOLD=1`:

| bundle | size | prompt | decode | gate |
|---|---:|---:|---:|---|
| int8lin | 3.2 GB | 125.4 | 107.4 | PASS 16/16 |
| int8hu --head-sym | 3.4 GB | 139.5 | 116.7 | PASS 16/16 |
| int4lin | 2.0 GB | 170.6 | 139.2 | PASS 16/16 |

Against the 1.2B's 276.5 tok/s: 2.2× the parameters, 2.4× slower — plain bandwidth scaling, no
surprise term. No iPhone measurement has been taken.

## Runtime notes

It is a **reasoning model**: the chat template ends the generation prompt with an open `<think>`,
and generations spend their first few hundred tokens thinking. A 200-token cap regularly ends
mid-thought. (The VL siblings are *not* thinking models — their generation prompt does not open
`<think>` — so do not copy this assumption sideways within the family.)

`llm-runner` needs `--warmup off` on these static-S=1 bundles unless the runtime carries the
warmup patch: default warmup submits a synthetic 256-token prefill and dies with

```
NDArrayDescriptor.swift:139: Fatal error: Shape at dimension 1 of 256 is not a valid substitution for source shape 1
```

before generating anything. Same for `coreai_gate.py`, which disables warmup deliberately.
