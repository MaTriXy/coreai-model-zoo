# Nanbeige4.2-3B (text decoder) — Core AI

[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) is a recurrent Llama decoder:
**22 physical transformer blocks execute twice**, with an RMS norm after each pass and separate KV history.
The Core AI model therefore executes 44 layer passes and uses **44 KV-cache layers**, while storing and
quantizing only one copy of the 22 physical blocks.

Source checkpoint: `5ff54fb7ed86ce8e216d78bff5417ab9981de3d4` (Apache-2.0).
Ported by [Vadim Smirnov (@ukint-vs)](https://github.com/ukint-vs); tracked in
[`john-rocky/coreai-model-zoo#5`](https://github.com/john-rocky/coreai-model-zoo/issues/5).

**⬇️ Verified int8 LanguageBundle:
[ukint-vs/Nanbeige4.2-3B-CoreAI](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e)**

- immutable bundle revision: `5864ec7a5581940958e58354a6b6c46c8f06891e`
- path: `gpu-pipelined/nanbeige4_2_3b_decode_int8hu_block32_sym_s1`
- format: `int8hu --head-sym --static-ids`, complete tokenizer included
- producer: `coreai-core 1.0.0b2`

## Runtime contract

The bundle uses the existing `coreai-pipelined` static-S=1 GPU path:

```text
input_ids, position_ids, k_cache, v_cache → logits
```

K/V states have shape `[44, 1, 8, max_context, 128]`. Pass 0 uses cache slots 0–21 and pass 1 uses 22–43.
Quantization visits 111 physical linear modules; recurrent execution does not duplicate serialized weights.
The bundled vendor chat template supports both `enable_thinking=true` and `enable_thinking=false`.

## Verified

| Check | Result |
|---|---|
| Bundle | **4.59 GiB** (`4,815,288 KiB`), int8 |
| Float32 parity | Full and cached logits pass at `rtol=1e-4`, `atol=1e-4`; identical 32-token greedy continuation |
| Int8 authoring gate | Prompt top-1 8/8, greedy 32/32, cosine 0.9997768 |
| Int8 engine gate | Token-exact vs fp32 on two factual prompts and the `9.11` versus `9.8` reasoning smoke; deterministic rerun |
| M4 Max Release, prompt 128 / generation 256 / 3 runs | **47.37 prefill / 46.35 decode tok/s** |
| Maximum context tested | **4,096 tokens**: 29.83 prefill / 32.80 decode tok/s, 9.17 GiB peak RSS, zero swaps |
| iOS 27 GPU AOT `h18p` | Compile pass with Xcode 27; generated package source hash matches the published model |
| iPhone `h18p` hardware | Pending; no iPhone throughput or memory claim |

The Mac benchmark used macOS 27.0, Xcode 27 beta 4, Core AI runtime `aff0bb2`, AC power, and High Power Mode.
The checkpoint config SHA-256 is
`f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`.

Int4 and mixed int4/int8 candidates are not published because they failed the same multi-token quality gates
required of int8. Detailed architecture, parity, quantization, and optimization results are in the
[support report](../knowledge/nanbeige4.2-coreai-support.md).

## Convert and verify

From a checkout at the commit pinned in `conversion/overlay/BASE`, apply the Python overlay, then run:

```sh
python3 ../coreai-model-zoo/conversion/zoo_convert.py run nanbeige4.2-3b --dry-run
python3 ../coreai-model-zoo/conversion/export_nanbeige41_decode_pipelined.py \
  int8hu --head-sym --static-ids \
  --hf-id Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4

python3 ../coreai-model-zoo/conversion/coreai_gate.py \
  exports/nanbeige4_2_3b_decode_int8hu_block32_sym_s1 \
  Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4 \
  --arch nanbeige -n 24
```

Reproduce the isolated official-checkpoint parity gate with:

```sh
python3 ../coreai-model-zoo/_smoke/verify_nanbeige42_checkpoint.py \
  --official-python /path/to/nanbeige-oracle/bin/python
```

The advertised 262K context is not claimed; the published bundle is verified to 4K. CoreAIKit enrollment and
the generated “Use it” block remain pending until the iOS 27 `h18p` device gate passes.

## License

Model weights: **Apache-2.0** (Nanbeige LLM Lab). Conversion code: BSD-3-Clause.
