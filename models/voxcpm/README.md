# VoxCPM-0.5B

> **Stub card.** This port is published and downloadable, but its full card has not
> been written yet. The authority for now is the model page itself:
> [🤗 mlboydaisuke/VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI). Everything below is read from the
> published artifact, not transcribed by hand.

## Published artifact

- **Repo**: [mlboydaisuke/VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI)
- **Bundles** (14):

  - `ios/voxcpm_base_int8_decode_cl512.h18p.aimodelc`
  - `ios/voxcpm_base_int8_prefill_t32.h18p.aimodelc`
  - `ios/voxcpm_feat_decoder_fp16.h18p.aimodelc`
  - `ios/voxcpm_feat_encoder_fp16.h18p.aimodelc`
  - `ios/voxcpm_res_int8_decode_cl512.h18p.aimodelc`
  - `ios/voxcpm_res_int8_prefill_t32.h18p.aimodelc`
  - `ios/voxcpm_vocoder_fp16_t12.h18p.aimodelc`
  - `macos/voxcpm_base_int8_decode_cl512/voxcpm_base_int8_decode_cl512.aimodel`
  - `macos/voxcpm_base_int8_prefill_t32/voxcpm_base_int8_prefill_t32.aimodel`
  - `macos/voxcpm_feat_decoder_fp16/voxcpm_feat_decoder_fp16.aimodel`
  - `macos/voxcpm_feat_encoder_fp16/voxcpm_feat_encoder_fp16.aimodel`
  - `macos/voxcpm_res_int8_decode_cl512/voxcpm_res_int8_decode_cl512.aimodel`
  - `macos/voxcpm_res_int8_prefill_t32/voxcpm_res_int8_prefill_t32.aimodel`
  - `macos/voxcpm_vocoder_fp16_t12/voxcpm_vocoder_fp16_t12.aimodel`

## Reproducing it

Conversion code: [`conversion/voxcpm/`](../../conversion/voxcpm/).
Port notes: [`knowledge/voxcpm-tts.md`](../../knowledge/voxcpm-tts.md).

No `recipe.toml` yet: the configuration that produced these bundles is not recorded,
and this catalog does not guess one. `python3 conversion/zoo_verify.py mlboydaisuke/VoxCPM-0.5B-CoreAI`
checks what *is* published (tokenizer, chat template, context, precision) against the
source model above.
