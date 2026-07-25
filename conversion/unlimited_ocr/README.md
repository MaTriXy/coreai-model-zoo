# Unlimited-OCR → Core AI conversion

Port of [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (3B-A0.5B MoE doc-OCR,
MIT) to Core AI: a fp16 DeepEncoder vision `.aimodel` + a sym8 DeepseekV2 **R-SWA** MoE decoder
(unified `prefill`+`decode` bundle) driven on the **stock `coreai.runtime`** (no engine patch). See
[`../../models/unlimited-ocr/README.md`](../../models/unlimited-ocr/README.md) and
[`../../knowledge/unlimited-ocr-rswa-static-decode.md`](../../knowledge/unlimited-ocr-rswa-static-decode.md).

## Files

| file | what |
|---|---|
| `model.py` | the decoder authored on Core AI macOS primitives (plain MHA + R-SWA mask + data-driven static-shape decode + greedy-softmax MoE); loader + prefill/decode/static-decode graphs |
| `export_vision.py` | DeepEncoder → `.aimodel` + self-gate vs the fp32 oracle (cos 1.0) |
| `export_decoder.py` | metalize MoE (sym8) → export prefill+decode → engine gate vs oracle; `--unified` = one bundle, `--no-metal` / `--max-step` / `--layers` for debugging |
| `arrange_assets.py` | verify the visual-token arrangement reconstructs the oracle prefix (cos 1.0) **and** write the Swift/host assets (`embed_tokens.f16`, `image_newline.f16`, `view_seperator.f16`, `prompt_input_ids.i32`, `recipe.json`) |
| `pipeline.py` | full image→markdown end-to-end via the engine (the app's reference) |
| `generate.py` | autoregressive generation from the oracle prefix (no_repeat_ngram) |
| `parity.py` | eager (no-engine) parity of the primitive decoder vs the oracle |
| `make_oracle.py` / `make_vision_oracle.py` | drive the HF model to dump the decoder / vision oracles + the R-SWA ≡ ring equivalence proof |

## Environments (two venvs)

- **oracle / HF load** — a dedicated venv with **`transformers==4.46.3`** + torch (the remote code
  targets 4.46.3; 5.x breaks it). torch-MPS has an immediate-EOS bug → run the oracle on **CPU fp32**.
- **export** — the Core AI authoring env (`coreai-torch` + `coreai.runtime`), py3.11.

## Steps

```sh
# 0) download the checkpoint into ./ckpt (HF baidu/Unlimited-OCR)
# 1) oracles (transformers==4.46.3 venv, CPU fp32)
python make_oracle.py            # -> out/_oracle/{oracle_tensors.npz, equiv_report.json}
python make_vision_oracle.py     # -> out/_vision_oracle/vision_tensors.npz

# 2) export + gate (Core AI env)
python export_vision.py                       # vision .aimodel, cos 1.0
python parity.py                              # eager decoder parity (cos 1.0 / 0 flips)
python export_decoder.py --unified            # prefill+decode unified bundle, engine gate (0 flips)

# 3) host assets + end-to-end check
python arrange_assets.py                      # -> out/_swift_assets/* (embed table + constants + recipe)
python pipeline.py --no-repeat-ngram 35       # image -> markdown via the engine
```

Bundles + assets are published at
[mlboydaisuke/Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI); the
macOS app is [`apps/CoreAIOCR`](../../apps/CoreAIOCR).
