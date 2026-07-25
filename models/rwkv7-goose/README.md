# RWKV7-Goose-1.5B

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/RWKV7-Goose-1.5B-CoreAI](https://huggingface.co/mlboydaisuke/RWKV7-Goose-1.5B-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/RWKV7-Goose-1.5B-CoreAI](https://huggingface.co/mlboydaisuke/RWKV7-Goose-1.5B-CoreAI)
- **Bundles** (2):

  - `aimodel/rwkv7_goose_1_5b/rwkv7_goose_1_5b.aimodel`
  - `h18p/rwkv7_goose_1_5b/rwkv7_goose_1_5b.h18p.aimodelc`

## Reproducing it

Conversion code: [`conversion/export_rwkv7_decode.py`](../../conversion/export_rwkv7_decode.py).
Port notes: [`knowledge/rwkv7-recurrent-linear-attention-coreai.md`](../../knowledge/rwkv7-recurrent-linear-attention-coreai.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/RWKV7-Goose-1.5B-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
