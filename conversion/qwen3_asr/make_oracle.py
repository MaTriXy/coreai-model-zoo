"""P0 oracle: run the official Qwen3-ASR on known-text clips → golden transcripts.

Uses the upstream `qwen_asr` package (added to sys.path, not installed) on CPU/fp32 so the
result is the deterministic reference the P1 eager re-implementation must match token-for-token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OFFICIAL = "/tmp/qwen3-asr-official"
sys.path.insert(0, OFFICIAL)

import torch  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
CLIPS = {
    "en1": "/tmp/qwen3asr_audio/en1.wav",  # "The quick brown fox jumps over the lazy dog."
    "en2": "/tmp/qwen3asr_audio/en2.wav",  # "Core AI runs large language models on device."
    "ja1": "/tmp/qwen3asr_audio/ja1.wav",  # Japanese weather sentence
}
OUT = Path(__file__).resolve().parent / "oracle_golden.json"


def main() -> None:
    print(f"loading {MODEL} (cpu / fp32, no forced aligner) ...", flush=True)
    asr = Qwen3ASRModel.from_pretrained(
        MODEL,
        dtype=torch.float32,
        device_map="cpu",
        forced_aligner=None,
        max_new_tokens=128,
    )
    print("loaded.", flush=True)

    golden = {}
    for name, path in CLIPS.items():
        res = asr.transcribe(audio=path, language=None, return_time_stamps=False)
        r = res[0]
        print(f"[{name}] language={r.language!r}\n[{name}] text={r.text!r}", flush=True)
        golden[name] = {"path": path, "language": r.language, "text": r.text}

    OUT.write_text(json.dumps(golden, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
