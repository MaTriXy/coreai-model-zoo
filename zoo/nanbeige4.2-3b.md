# Nanbeige4.2-3B (text decoder) — Core AI

Native recurrent-Llama authoring for the released
[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) checkpoint at immutable
revision `5ff54fb7ed86ce8e216d78bff5417ab9981de3d4` (Apache-2.0). The checkpoint has **22 physical
transformer layers** whose weights run twice; the exported decoder therefore executes 44 layer passes and keeps
**44 independent KV cache layers** without duplicating the 22 trainable blocks.

The complete architecture, conversion, parity, quantization, runtime, hardware, and kernel investigation is in
the [Nanbeige4.2 Core AI support report](../knowledge/nanbeige4.2-coreai-support.md).

## Release status

**Conversion support is implemented; publication is pending.** No artifact has been uploaded and no CoreAIKit
catalog entry has been written. The shipping candidate is `int8hu --head-sym --static-ids`. The matching
`int4hu` candidate was measured and is a **quality no-go**; it must not replace int8.

| Required measurement | Result |
|---|---|
| Bundle size | **4.59 GiB on disk** (`4,815,288 KiB`; `.aimodel` 4.57 GiB), int8hu baseline |
| Float32 full/incremental parity | **Pass:** full and cached logits allclose at `rtol=1e-4`, `atol=1e-4`; identical 32-token greedy continuation |
| Int8 authoring/oracle gate | **Pass:** prompt top-1 8/8, greedy 32/32, cosine 0.9997768, deterministic |
| Int8 Core AI engine gate | **Pass:** token-exact vs fp32 for Paris (24 tokens), 0°C (16), and the 64-token-max reasoning smoke concluding 9.8 > 9.11; deterministic rerun |
| Int4 candidate | **No-go:** 3.14 GiB and deterministic, but Paris diverged after 2/24 tokens, freezing after 3/16, and reasoning failed decisively at token 0 (`margin=0.8237`) |
| Mixed int4/int8 candidate | **No-go:** layers 3/4/5 in int4 reduced the bundle to 4.41 GiB, but the Core AI reasoning gate failed decisively at token 0 (`margin=0.8542`); no mixed mode is exposed |
| Core AI bundle load/cache smoke | **Pass:** load 2.34 s; one token produced logits and mutated all 44 cache layers |
| Mac M4 Max, prompt 128 / generation 256 / 3 runs | **Pass:** 47.37 prefill / 46.35 decode tok/s average |
| iPhone Release/AOT `h18p`, same workload | **Blocked:** connected iPhone 16 Pro is `h17p` on iOS 26.6; Xcode 27 cannot mount its developer image |
| Peak memory and maximum context actually tested | **9.17 GiB peak RSS**, zero swaps; full 4,096-token boundary passed at 29.83 prefill / 32.80 decode tok/s |

The upstream config bytes are pinned by SHA-256
`f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`. The advertised 262K context is not
claimed here; the recipe defaults to a 4K export until a larger context is measured on both target devices.

The committed isolated official-checkpoint versus Core AI overlay gate reports `1.01566e-4` full and
`2.09808e-5` incremental maximum absolute error at `rtol=1e-4`, `atol=1e-4`, with the same 32-token greedy
continuation. The measured int8 quality results use the exact shipping quantization traversal (111 physical
linear modules). The Release Core AI bundle on runtime
`aff0bb2` then matched its pinned fp32 authoring oracle token-for-token for all three public prompts. The pinned
tokenizer does not contain the configurable vendor chat template described by its model README, so a separately
verified chat-template integration remains a publication gate.

The accepted M4 Max benchmark used macOS 27.0 (`26A5378n`), Xcode 27 beta 4, Core AI runtime `aff0bb2`,
static-S=1, AC power, High Power Mode, prompt 128, generation 256, and three trials. Prefill was 48.84, 48.45,
and 44.82 tok/s; decode was 46.74, 46.88, and 45.43 tok/s. The system reported no thermal or performance
warning. The full exported boundary (3,840 prompt + 256 generation) completed with 9.17 GiB maximum RSS and no
swaps. A discarded run with battery saving enabled averaged only 21.83 decode tok/s. The static-output capacity
and descriptor-driven single-token prefill/warmup fixes in `apps/coreai-pipelined-extra-states.patch` are
required.

The connected iPhone 16 Pro (`iPhone17,1`) runs iOS 26.6, so Xcode 27 cannot mount a compatible developer image
and Core AI device execution is unavailable. It also targets `h17p`, while this release criterion requires
`h18p`. iPhone acceptance therefore remains failed—not extrapolated from Mac results—until an iOS 27 `h18p`
device is available.

For comparison only, the int4 candidate occupies 3.14 GiB and peaks at 6.24 GiB RSS with zero swaps. Its first
128/256 run was still rising (38.01 prefill / 37.74 decode tok/s average); after the full-context warmup, the
three stable trials averaged 58.91 prefill / 56.07 decode tok/s. The 4,096-token boundary passed at 45.47 prefill
/ 44.46 decode tok/s. These speed and memory gains do not override the failed quality gate.

A physical-layer mixed-precision sweep also failed. Twelve of 22 layers were individually harmless under the
eager margin gate, but combining them accumulated decisive errors. The largest multi-layer survivors were
exported: the four-layer (`3/4/5/10`) and three-layer (`3/4/5`) candidates both selected a different
high-margin reasoning branch in Core AI. The smaller bundle was 4.41 GiB. It was not benchmarked because quality
gates precede performance acceptance, and no experimental mixed mode remains in the conversion interface.

## Convert

From a `coreai-models` checkout with this repository's Python overlay applied:

```sh
python3 ../coreai-model-zoo/conversion/zoo_convert.py run nanbeige4.2-3b --dry-run
python3 ../coreai-model-zoo/conversion/export_nanbeige41_decode_pipelined.py \
  int8hu --head-sym --static-ids \
  --hf-id Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4
```

Reproduce the pinned official-vs-overlay float32 gate with the overlay interpreter and a separate
vendor-compatible environment:

```sh
python3 ../coreai-model-zoo/_smoke/verify_nanbeige42_checkpoint.py \
  --official-python /path/to/nanbeige-oracle/bin/python
```

Gate an exported bundle through the Release runtime:

```sh
python3 ../coreai-model-zoo/conversion/coreai_gate.py \
  exports/nanbeige4_2_3b_decode_int8hu_block32_sym_s1 \
  Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4 --arch nanbeige -n 24
```

The bundle interface remains `input_ids`, `position_ids`, mutable `k_cache`, `v_cache` → `logits`; cache tensors
have shape `[44, 1, 8, max_context, 128]`. The existing static-S=1 pipelined runtime path is used on device.

## Remaining acceptance gates before publication

- Run the same Release benchmark and 24-token oracle gate on an iOS 27 `h18p` iPhone with the increased-memory
  entitlement. The evaluated int4 candidate has failed and cannot be used as a fallback.
- Supply and verify the intended chat template; the pinned tokenizer has none.

After those results are recorded and publication is separately approved, upload the selected immutable bundle,
add `nanbeige4.2-3b` to CoreAIKit with `kind: chat`, `engine: pipelined`, `thinking: true`, enroll the card in
`cards.json`, and regenerate the managed “Use it” block.
