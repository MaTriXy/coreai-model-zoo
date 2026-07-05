"""Phase 1 — Nemotron 3.5 ASR streaming 0.6B golden oracle (run in the ISOLATED tf-source env).

Loads nvidia/nemotron-3.5-asr-streaming-0.6b, runs the OFFLINE (full-utterance) path on a real
clip with a fixed language prompt, and saves a golden bundle the Core-AI export/gate compares
against (same idiom as ../parakeet/gen_oracle.py):

  oracle_<lang>.npz
    input_features   [1,L,128]     log-mel (processor layout)
    enc_last         [T,H]         raw FastConformer output (pre prompt-fusion)
    enc_proj         [T,P]         encoder_projector(prompt_projector fusion) — what the joint consumes
    prompt_id        ()            language prompt index (ja-JP=10, en-US=0, auto=101)
    one_hot          [num_prompts] the broadcast prompt vector (constant per language)
    tokens           [U]           emitted token ids (no start token)
    step_frames/step_tokens/step_durations/step_logits   greedy-loop trace
    text             ()            reference transcript (str)
  + scalars: blank_id, start_token, vocab_size, n_durations, T_valid, num_lookahead_tokens

Also asserts our hand-rolled greedy transducer loop == model.generate() (the loop we port to
Swift). Duration slots are auto-detected: if the joint head emits more logits than the token
vocab, the trailing len(config.durations) are TDT duration logits; otherwise pure RNNT
(blank advances one frame, max_symbols_per_step caps emissions per frame).

Run:  ~/code/coreai/_nemotron_oracle/.venv/bin/python gen_oracle.py [--language ja-JP] [--wav clip.wav]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
HERE = Path(__file__).resolve().parent


def load_clip(wav_path: str | None, seconds: float, sr: int = 16000) -> np.ndarray:
    import librosa
    if wav_path:
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
    else:
        wav, _ = librosa.load(librosa.example("libri1"), sr=sr, mono=True)
    return wav[: int(seconds * sr)]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="en-US", help="prompt language (en-US / ja-JP / auto ...)")
    ap.add_argument("--wav", default=None, help="optional wav/flac path (default: librosa libri1 clip)")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = HERE / (args.out or f"oracle_{args.language.replace('-', '_')}.npz")

    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

    processor = AutoProcessor.from_pretrained(MODEL)
    model = Nemotron3_5AsrForRNNT.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config
    blank = cfg.blank_token_id
    tok_vocab = len(processor.tokenizer)
    durations = list(getattr(cfg, "durations", []) or [])
    start = getattr(model.generation_config, "decoder_start_token_id", None)
    if start is None:
        start = blank
    from transformers.activations import ACT2FN
    joint_act = ACT2FN[cfg.hidden_act]
    head_out = model.joint.head.out_features
    is_tdt = bool(durations) and head_out > tok_vocab
    print(f"blank={blank} tokenizer_vocab={tok_vocab} head_out={head_out} "
          f"durations={durations} tdt={is_tdt} start={start} act={cfg.hidden_act} "
          f"num_prompts={cfg.num_prompts} default_prompt={cfg.default_prompt_id}")

    wav = load_clip(args.wav, args.seconds)
    inputs = processor(wav, sampling_rate=16000, language=args.language, return_tensors="pt")
    feats = inputs["input_features"]
    prompt_ids = inputs["prompt_ids"]
    lookahead = int(inputs.get("num_lookahead_tokens", cfg.encoder_config.default_num_lookahead_tokens))
    print(f"input_features {tuple(feats.shape)} prompt_id={int(prompt_ids[0])} lookahead={lookahead}")

    # --- reference transcript via the official generate() ---
    gen = model.generate(input_features=feats, prompt_ids=prompt_ids,
                         num_lookahead_tokens=lookahead)
    dur_kw = {"durations": gen.durations} if getattr(gen, "durations", None) is not None else {}
    ref_text = processor.batch_decode(gen.sequences, **dur_kw)[0]
    ref_tokens = [t for t in gen.sequences[0].tolist() if t != start and t != blank]
    print(f"[generate] text: {ref_text!r}")

    # --- encoder (pre-fusion) + prompt fusion + projector ---
    enc = model.get_audio_features(input_features=feats, prompt_ids=prompt_ids,
                                   num_lookahead_tokens=lookahead)
    enc_last = enc.last_hidden_state[0]          # [T,H] pre prompt-fusion
    enc_proj = enc.pooler_output[0]              # [T,P] post fusion+projector
    T = enc_proj.shape[0]
    one_hot = torch.nn.functional.one_hot(prompt_ids[0], num_classes=cfg.num_prompts).float()
    print(f"encoder out: last {tuple(enc_last.shape)} proj {tuple(enc_proj.shape)} T={T}")

    # --- hand-rolled greedy transducer loop (the algorithm we port to the host) ---
    dec = model.decoder

    def decode_step(token_id: int, state):
        emb = dec.embedding(torch.tensor([[token_id]]))
        lstm_out, new_state = dec.lstm(emb, state)
        return dec.decoder_projector(lstm_out)[0, 0], new_state

    h = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size)
    c = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size)
    dec_out, (h, c) = decode_step(start, (h, c))

    frame, symbols_on_frame = 0, 0
    emitted: list[int] = []
    step_frames, step_tokens, step_durs, step_logits = [], [], [], []
    max_steps = cfg.max_symbols_per_step * T + 16
    while frame < T and len(step_tokens) < max_steps:
        logits = model.joint.head(joint_act(enc_proj[frame] + dec_out))
        token = int(logits[:tok_vocab].argmax())
        if is_tdt:
            dur = durations[int(logits[tok_vocab:].argmax())]
            if token == blank and dur == 0:
                dur = 1
        else:  # pure RNNT: blank (or symbol cap) advances one frame
            symbols_on_frame = 0 if token == blank else symbols_on_frame + 1
            dur = 1 if (token == blank or symbols_on_frame >= cfg.max_symbols_per_step) else 0
            if dur == 1:
                symbols_on_frame = 0
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
    print(f"[check] hand-loop tokens == generate(): {match} (mine {len(emitted)} vs ref {len(ref_tokens)})")

    np.savez_compressed(
        out,
        input_features=feats.numpy().astype(np.float32),
        enc_last=enc_last.numpy().astype(np.float32),
        enc_proj=enc_proj.numpy().astype(np.float32),
        prompt_id=np.array(int(prompt_ids[0])),
        one_hot=one_hot.numpy().astype(np.float32),
        tokens=np.array(emitted, dtype=np.int64),
        step_frames=np.array(step_frames, dtype=np.int64),
        step_tokens=np.array(step_tokens, dtype=np.int64),
        step_durations=np.array(step_durs, dtype=np.int64),
        step_logits=np.stack(step_logits).astype(np.float32),
        text=np.array(ref_text),
        blank_id=np.array(blank), start_token=np.array(start),
        vocab_size=np.array(tok_vocab), n_durations=np.array(len(durations)),
        T_valid=np.array(T), num_lookahead_tokens=np.array(lookahead),
    )
    print(f"[save] {out} ({out.stat().st_size / 1e6:.1f} MB)  "
          f"{'✅ loop matches generate()' if match else '❌ MISMATCH — fix loop before export'}")


if __name__ == "__main__":
    main()
