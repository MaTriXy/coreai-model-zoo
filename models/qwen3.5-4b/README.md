# qwen3.5-4B

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/qwen3.5-4B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-4B-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/qwen3.5-4B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-4B-CoreAI)
- **Source model**: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Bundles** (1):

  - `gpu-pipelined-b2/qwen3_5_4b_decode_int8hu_block32_sym`

## Reproducing it

Conversion code: [`conversion/export_qwen3_5_decode_pipelined.py`](../../conversion/export_qwen3_5_decode_pipelined.py).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/qwen3.5-4B-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
