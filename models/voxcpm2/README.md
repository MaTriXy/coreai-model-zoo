# VoxCPM2

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI)
- **Bundles** (14):

  - `ios/voxcpm2_base_int8_decode_cl512.h18p.aimodelc`
  - `ios/voxcpm2_base_int8_prefill_t32.h18p.aimodelc`
  - `ios/voxcpm2_feat_decoder_fp16.h18p.aimodelc`
  - `ios/voxcpm2_feat_encoder_fp16.h18p.aimodelc`
  - `ios/voxcpm2_res_int8_decode_cl512.h18p.aimodelc`
  - `ios/voxcpm2_res_int8_prefill_t32.h18p.aimodelc`
  - `ios/voxcpm2_vocoder_fp16_t8.h18p.aimodelc`
  - `macos/voxcpm2_base_int8_decode_cl512/voxcpm2_base_int8_decode_cl512.aimodel`
  - `macos/voxcpm2_base_int8_prefill_t32/voxcpm2_base_int8_prefill_t32.aimodel`
  - `macos/voxcpm2_feat_decoder_fp16/voxcpm2_feat_decoder_fp16.aimodel`
  - `macos/voxcpm2_feat_encoder_fp16/voxcpm2_feat_encoder_fp16.aimodel`
  - `macos/voxcpm2_res_int8_decode_cl512/voxcpm2_res_int8_decode_cl512.aimodel`
  - `macos/voxcpm2_res_int8_prefill_t32/voxcpm2_res_int8_prefill_t32.aimodel`
  - `macos/voxcpm2_vocoder_fp16_t8/voxcpm2_vocoder_fp16_t8.aimodel`

## Reproducing it

Conversion code: [`conversion/voxcpm/`](../../conversion/voxcpm/).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/VoxCPM2-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
