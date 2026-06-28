"""Phase 1 — Parakeet TDT 0.6B golden oracle (run in the ISOLATED transformers-5.x env).

Loads nvidia/parakeet-tdt-0.6b-v3, runs it on a real English clip, and saves a golden
bundle that the Core-AI export/gate (main coreai-torch venv) compares against:

  oracle.npz
    input_features   [1,128,Tm]   log-mel (the encoder bundle's input)
    enc_last         [T,1024]      raw FastConformer output
    enc_proj         [T,640]       encoder_projector(enc_last) — what the joint consumes
    tokens           [U]           emitted token ids (no start token)
    step_frames      [S]           encoder frame index at each decode step
    step_tokens      [S]           argmax token at each step (incl. blanks)
    step_durations   [S]           argmax duration at each step
    step_logits      [S,8198]      reference joint logits (token 0..8192 + dur 8193..8197)
    text             ()            reference transcript (str)
  + scalars: blank_id, start_token, vocab_size, n_durations, T_valid

Also asserts our hand-rolled greedy TDT loop == model.generate() (the loop we port to Swift).

Run:  ~/code/coreai/_parakeet_oracle/.venv/bin/python gen_oracle.py [--seconds 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
OUT = Path(__file__).resolve().parent / "oracle.npz"


def load_clip(seconds: float, pad_seconds: float = 0.0, sr: int = 16000) -> np.ndarray:
    import librosa
    path = librosa.example("libri1")  # a real LibriSpeech English utterance
    wav, _ = librosa.load(path, sr=sr, mono=True)
    wav = wav[: int(seconds * sr)]
    if pad_seconds > 0:  # trailing silence -> the app's fixed-bucket scenario (no mask)
        wav = np.concatenate([wav, np.zeros(int(pad_seconds * sr), dtype=wav.dtype)])
    return wav


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--pad-seconds", type=float, default=0.0, help="trailing silence to a fixed bucket")
    ap.add_argument("--out", default="oracle.npz")
    args = ap.parse_args()
    global OUT
    OUT = Path(__file__).resolve().parent / args.out

    from transformers import AutoProcessor, ParakeetForTDT

    processor = AutoProcessor.from_pretrained(MODEL)
    model = ParakeetForTDT.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config
    blank = cfg.blank_token_id
    vocab = cfg.vocab_size
    durations = list(cfg.durations)
    start = model.generation_config.decoder_start_token_id
    if start is None:
        start = blank
    # The joint network's activation is the TOP-level config.hidden_act (ReLU here),
    # NOT the encoder's silu — getting this wrong silently garbles the transcript.
    joint_act = torch.nn.functional.relu if cfg.hidden_act == "relu" else torch.nn.functional.silu
    print(f"blank={blank} vocab={vocab} durations={durations} start_token={start} joint_act={cfg.hidden_act}")

    wav = load_clip(args.seconds, args.pad_seconds)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"]
    print(f"input_features {tuple(feats.shape)}")

    # --- reference transcript via the official generate() ---
    gen = model.generate(input_features=feats)
    ref_text = processor.batch_decode(gen.sequences, durations=gen.durations)[0]
    ref_tokens = [t for t in gen.sequences[0].tolist() if t != start and t != blank]
    print(f"[generate] text: {ref_text!r}")

    # --- encoder + projector ---
    enc = model.get_audio_features(input_features=feats)
    enc_last = enc.last_hidden_state[0]          # [T,1024]
    enc_proj = enc.pooler_output[0]              # [T,640]
    T = enc_proj.shape[0]
    print(f"encoder out: last {tuple(enc_last.shape)} proj {tuple(enc_proj.shape)}  T={T}")

    # --- hand-rolled greedy TDT loop (the algorithm we port to the host) ---
    dec = model.decoder
    head = model.joint.head

    def decode_step(token_id: int, state):
        emb = dec.embedding(torch.tensor([[token_id]]))
        out, new_state = dec.lstm(emb, state)
        return dec.decoder_projector(out)[0, 0], new_state    # [640], (h,c)

    h = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size)
    c = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size)
    dec_out, (h, c) = decode_step(start, (h, c))

    frame = 0
    emitted: list[int] = []
    step_frames, step_tokens, step_durs, step_logits = [], [], [], []
    max_steps = cfg.max_symbols_per_step * T + 16
    while frame < T and len(step_tokens) < max_steps:
        logits = head(joint_act(enc_proj[frame] + dec_out))     # [8198]
        token = int(logits[:vocab].argmax())
        dur_idx = int(logits[vocab:].argmax())
        dur = durations[dur_idx]
        if token == blank and dur == 0:
            dur = 1
        step_frames.append(frame)
        step_tokens.append(token)
        step_durs.append(dur)
        step_logits.append(logits.numpy())
        frame += dur
        if token != blank:
            emitted.append(token)
            dec_out, (h, c) = decode_step(token, (h, c))

    my_text = processor.tokenizer.decode(emitted)
    print(f"[hand-loop] steps={len(step_tokens)} emitted={len(emitted)} text: {my_text!r}")

    match = emitted == ref_tokens
    print(f"[check] hand-loop tokens == generate(): {match}  "
          f"(mine {len(emitted)} vs ref {len(ref_tokens)})")

    np.savez_compressed(
        OUT,
        input_features=feats.numpy().astype(np.float32),
        enc_last=enc_last.numpy().astype(np.float32),
        enc_proj=enc_proj.numpy().astype(np.float32),
        tokens=np.array(emitted, dtype=np.int64),
        step_frames=np.array(step_frames, dtype=np.int64),
        step_tokens=np.array(step_tokens, dtype=np.int64),
        step_durations=np.array(step_durs, dtype=np.int64),
        step_logits=np.stack(step_logits).astype(np.float32),
        text=np.array(ref_text),
        blank_id=np.array(blank), start_token=np.array(start),
        vocab_size=np.array(vocab), n_durations=np.array(len(durations)),
        T_valid=np.array(T),
    )
    sz = OUT.stat().st_size / 1e6
    print(f"[save] {OUT} ({sz:.1f} MB)  {'✅ loop matches generate()' if match else '❌ MISMATCH'}")


if __name__ == "__main__":
    main()
