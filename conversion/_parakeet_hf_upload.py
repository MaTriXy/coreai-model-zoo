"""Stage + upload the Parakeet-TDT-0.6B Core AI bundle to HF. USER-GATED — run only when asked.

Stages the 30 s encoder (fp16) + predict/joint (fp32) .aimodels + tokenizer (with the
swift-transformers-compatible `tokenizer_class`) + the librosa-slaney mel filterbank + a cc-by-4.0
model card into /tmp/parakeet_hf, then uploads to mlboydaisuke/Parakeet-TDT-0.6B-CoreAI.

    coreai-models/.venv/bin/python conversion/_parakeet_hf_upload.py
"""
import os, json, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/Parakeet-TDT-0.6B-CoreAI"
HERE = Path(__file__).resolve().parent
ART = HERE / "parakeet" / "artifacts"
ASSETS = ART / "bundle_assets"
STAGE = Path("/tmp/parakeet_hf")

CARD = """---
license: cc-by-4.0
library_name: coreai
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt-0.6b-v3
tags: [core-ai, coreaikit, parakeet, tdt, rnn-t, transducer, asr, on-device, apple]
---

# Parakeet-TDT-0.6B — Core AI

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (cc-by-4.0, 600M)
converted to **Apple Core AI** `.aimodel` — the first **transducer / TDT (RNN-T family)** ASR in the
[zoo](https://github.com/john-rocky/coreai-models-community). Transcribes ≤~29 s clips in 25
European languages as **three stateless graphs + a host greedy loop** (no LLM runtime).

- `parakeet_encoder_float16_L2885.aimodel` — FastConformer encoder + projector (fp16, ~1.2 GB),
  `mel[1,128,2885] → enc_proj[1,361,640]`.
- `parakeet_predict_float32.aimodel` — embedding → 2-layer LSTM → projector (fp32),
  `token[1,1],h,c[2,1,640] → dec_out[1,640],h',c'`.
- `parakeet_joint_float32.aimodel` — `head(relu(enc_frame+dec_out))` (fp32),
  `→ token_logits[1,8193], dur_logits[1,5]`.
- `tokenizer.json` (+ `tokenizer_config.json`), `mel_filters_128x257_f32.bin` (librosa-slaney).

Gated **77/77 token-exact** end-to-end vs the HF `ParakeetForTDT` reference, and again token-exact
through the Swift **CoreAIKit** `KitParakeetModel`. blank 8192 · durations [0,1,2,3,4] · 16 kHz.

## Use (CoreAIKit)

```swift
let parakeet = try await KitParakeetModel(model: .parakeetTDT)
let result = try await parakeet.transcribe(samples: pcm16kMono)   // 16 kHz mono Float
```
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for name in ["parakeet_encoder_float16_L2885.aimodel", "parakeet_predict_float32.aimodel",
                 "parakeet_joint_float32.aimodel"]:
        shutil.copytree(ART / name, STAGE / name)
    for f in ["tokenizer.json", "mel_filters_128x257_f32.bin"]:
        shutil.copy2(ASSETS / f, STAGE / f)
    # swift-transformers (CoreAIKit) has no "ParakeetTokenizer"; retag to a registered class that
    # maps to BPETokenizer. Decode is driven by tokenizer.json's Metaspace decoder, so this is exact.
    cfg = json.loads((ASSETS / "tokenizer_config.json").read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizer"
    (STAGE / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
    (STAGE / "README.md").write_text(CARD)

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Parakeet-TDT-0.6B -> Core AI: encoder + predict + joint + tokenizer")
    print("uploaded", REPO)


if __name__ == "__main__":
    main()
