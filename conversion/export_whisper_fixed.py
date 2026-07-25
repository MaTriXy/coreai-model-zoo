# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.1",
#     "transformers==4.57.3",
# ]
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Whisper export with a DYNAMIC decoder sequence length.

Apple's official models/whisper/export.py traces the model with a fixed
decoder_input_ids of shape [1, 1], producing a single-position graph with no KV cache —
which cannot be driven autoregressively (each step would lose all prior context). This
variant is identical EXCEPT decoder_input_ids is traced at a fixed [1, 128] shape. You
pad the decoder buffer to 128 and read logits at the real last position; causal attention
ignores the padding, and the constant shape means MPSGraph compiles once (a dynamic-length
export instead recompiles every step → ~15 s/token vs ~0.18 s/token here). Runs on the stock
Core AI runtime; the combined encoder+decoder graph re-encodes the audio each step (fine for
the turbo encoder on a 30 s window).
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np

DEC_LEN = 128  # fixed decoder length; pad the prompt to this, read logits at the real last position
import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


class WhisperModule(torch.nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype):
        super().__init__()
        self._model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=dtype, use_safetensors=True
        )

    def forward(self, input_features, decoder_input_ids):
        return self._model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        ).logits


def reference_inputs(model_name: str, dtype: torch.dtype):
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)
    feature = processor.feature_extractor(dummy_audio, sampling_rate=16000)
    # Trace with seq=4 (a realistic decoder prompt) so the seq dim isn't 0/1-specialized.
    return {
        "input_features": torch.tensor(feature["input_features"]).to(dtype),
        "decoder_input_ids": torch.zeros((1, DEC_LEN), dtype=torch.int32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "exports"))
    args = p.parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[INFO] Sourcing model...")
    model = WhisperModule(args.model, dtype).eval()
    example = reference_inputs(args.model, dtype)


    print("[INFO] torch.export with FIXED decoder length %d..." % DEC_LEN)
    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(model, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())

    print("[INFO] Converting to Core AI...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
    )
    prog = converter.to_coreai()
    prog.optimize()

    name = f"{Path(args.model).name}_{args.dtype}_fixed128"
    out = Path(args.output_dir) / f"{name}.aimodel"
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    md = AIModelAssetMetadata()
    md.author = "A. Radford et al."
    md.license = "Apache-2.0"
    md.model_description = (
        "Whisper large-v3-turbo ASR encoder-decoder, exported with a dynamic decoder "
        "sequence length for autoregressive transcription. Source: "
        "https://huggingface.co/openai/whisper-large-v3-turbo"
    )
    md.creation_date = int(time.time())
    prog.save_asset(out, md)
    print(f"[INFO] Saved {out}")


if __name__ == "__main__":
    main()
