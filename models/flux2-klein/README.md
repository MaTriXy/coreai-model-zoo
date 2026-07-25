# FLUX.2-klein-4B

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)
- **Bundles** (7):

  - `TextEncoder.aimodel`
  - `Transformer.aimodel`
  - `Transformer_edit.aimodel`
  - `Transformer_edit_2ref.aimodel`
  - `Transformer_edit_512.aimodel`
  - `VAEDecoder.aimodel`
  - `VAEEncoder.aimodel`

## Reproducing it

The conversion scripts for this port are **not in this repository** — see `models/_INVENTORY.md`, section "Needs owner input".
Port notes: [`knowledge/flux2-in-context-editing.md`](../../knowledge/flux2-in-context-editing.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/FLUX.2-klein-4B-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
