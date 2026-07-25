# Stable-Audio-Open-Small

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI)
- **Bundles** (1):

  - `macos`

## Reproducing it

Conversion code: [`conversion/stable_audio/`](../../conversion/stable_audio/).
Port notes: [`knowledge/music-generation-stable-audio.md`](../../knowledge/music-generation-stable-audio.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/Stable-Audio-Open-Small-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
