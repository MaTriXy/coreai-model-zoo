# AdcSR

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI)
- **Bundles** (1):

  - `adcsr_x4_float32.aimodel`

## Reproducing it

Conversion code: [`conversion/export_adcsr.py`](../../conversion/export_adcsr.py).
Port notes: [`knowledge/adcsr-super-resolution.md`](../../knowledge/adcsr-super-resolution.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/AdcSR-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
