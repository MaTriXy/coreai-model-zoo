"""P1 token-level oracle: exact prompt ids, generated ids, audio-token count N, and the encoder
golden tensor — the deterministic targets the eager re-implementation must match.

Uses the upstream transformers backend classes directly (CPU/fp32) on one clean clip (ja1).
Saves tensors to oracle_tokens.npz + a readable oracle_tokens.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OFFICIAL = "/tmp/qwen3-asr-official"
sys.path.insert(0, OFFICIAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import librosa  # noqa: E402
from qwen_asr.core.transformers_backend import (  # noqa: E402
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

MODEL = "Qwen/Qwen3-ASR-1.7B"
CLIP = "/tmp/qwen3asr_audio/ja1.wav"  # clean reference (oracle transcript was perfect)
AUDIO_TOKEN_ID = 151676
OUTDIR = Path(__file__).resolve().parent


@torch.no_grad()
def main() -> None:
    print("loading model + processor (cpu/fp32) ...", flush=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    proc = Qwen3ASRProcessor.from_pretrained(MODEL, fix_mistral_regex=True)

    wav, sr = librosa.load(CLIP, sr=16000, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    print(f"clip {CLIP}: {len(wav)/sr:.2f}s @ {sr}Hz", flush=True)

    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(f"prompt template string:\n{text!r}", flush=True)

    inputs = proc(text=[text], audio=[wav], return_tensors="pt", padding=True)
    print("processor output keys:", list(inputs.keys()), flush=True)
    input_ids = inputs["input_ids"]
    N = int((input_ids == AUDIO_TOKEN_ID).sum().item())
    print(f"input_ids shape {tuple(input_ids.shape)}  audio_token_count N={N}", flush=True)
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}", flush=True)

    # encoder golden (the novel part to reproduce)
    feat_mask = inputs.get("feature_attention_mask", None)
    audio_feature_lengths = inputs.get("audio_feature_lengths", None)
    enc = model.thinker.get_audio_features(
        inputs["input_features"],
        feature_attention_mask=feat_mask,
        audio_feature_lengths=audio_feature_lengths,
    )
    print(f"encoder last_hidden_state: {tuple(enc.shape)} {enc.dtype}", flush=True)
    assert enc.shape[0] == N, f"encoder tokens {enc.shape[0]} != N {N}"

    # greedy generation golden
    out = model.generate(**inputs, max_new_tokens=128)  # top-level sets return_dict_in_generate
    gen_ids = out.sequences[:, input_ids.shape[1]:][0]
    decoded = proc.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
    print(f"\ngen_ids ({len(gen_ids)}): {gen_ids.tolist()}", flush=True)
    print(f"decoded: {decoded!r}", flush=True)

    np.savez(
        OUTDIR / "oracle_tokens.npz",
        input_ids=input_ids.numpy(),
        gen_ids=gen_ids.numpy(),
        input_features=inputs["input_features"].numpy(),
        feature_attention_mask=(feat_mask.numpy() if feat_mask is not None else np.array([])),
        encoder_out=enc.float().numpy(),
    )
    (OUTDIR / "oracle_tokens.json").write_text(json.dumps({
        "clip": CLIP, "N_audio_tokens": N,
        "prompt_template_str": text,
        "input_ids_len": int(input_ids.shape[1]),
        "gen_ids": gen_ids.tolist(),
        "decoded": decoded,
        "encoder_out_shape": list(enc.shape),
        "processor_keys": list(inputs.keys()),
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUTDIR/'oracle_tokens.npz'} + .json", flush=True)


if __name__ == "__main__":
    main()
