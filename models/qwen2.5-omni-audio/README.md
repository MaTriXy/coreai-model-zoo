# Qwen2.5-Omni-3B-Audio

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI)
- **Source model**: [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- **Bundles** (3):

  - `gpu-pipelined/qwen2_5_omni_3b_audio_encoder_fp16_k15/qwen2_5_omni_3b_audio_encoder_fp16_k15.aimodel`
  - `gpu-pipelined/qwen2_5_omni_3b_thinker_int8lin_n750_s1`
  - `ios/qwen2_5_omni_3b_thinker_n750_ios`

## Reproducing it

The conversion scripts for this port are **not in this repository** — see `models/_INVENTORY.md`, section "Needs owner input".
Port notes: [`knowledge/qwen2.5-omni-audio-understanding.md`](../../knowledge/qwen2.5-omni-audio-understanding.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
