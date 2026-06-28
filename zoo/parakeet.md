# Parakeet-TDT-0.6B — Core AI

The zoo's **first transducer / TDT (RNN-T family) model** — a different ASR architecture from the
attention decoders (Whisper, Qwen3-ASR). [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(cc-by-4.0, 600M) transcribes ≤~29 s clips in **25 European languages**. It runs as **three
stateless `.aimodel` graphs + a host-driven greedy loop** — no LLM runtime, just CoreAIKit's
`GraphModel`.

A **token-and-duration transducer (TDT)**: a **FastConformer encoder** streams acoustic frames, an
**LSTM predictor** carries the text state, and a **joint network** decides, at each (frame, token)
pair, both the next token and **how many frames to skip** (the "duration" head — the TDT speedup
over vanilla RNN-T). Blanks advance time; non-blank tokens advance the predictor. Fulfills
apple/coreai-models issue **#7**.

## Pipeline

```
16 kHz mono ──(log-mel, host: preemphasis→STFT→slaney mel→log→per-utt norm)──▶ mel[1,128,2885]
  1. encoder.aimodel : mel[1,128,2885] f16 ─▶ enc_proj[1,361,640]   (FastConformer + 1024→640 projector)
  host greedy TDT loop over the 361 encoder frames:
  2. predict.aimodel : token[1,1] i32 · h,c[2,1,640] ─▶ dec_out[1,640] · h',c'   (embedding→2-LSTM→proj)
  3. joint.aimodel   : dec_out[1,640] · enc_frame[1,640] ─▶ token_logits[1,8193] · dur_logits[1,5]
     token = argmax(token_logits) ; dur = [0,1,2,3,4][argmax(dur_logits)]
     if token==blank(8192) && dur==0 : dur = 1     # forward-progress guard
     frame += dur ; if token!=blank : emit(token) and step the predictor (advance LSTM only on non-blank)
  host: detokenize the emitted ids (BPE + Metaspace) ─▶ transcript
```

The encoder is baked at a **fixed 30 s bucket (2885 mel frames → 361 encoder frames)**; the host
pads/trims every clip to it. Trailing silence just makes the loop emit blanks, so no attention/conv
masking is needed — the simple static graph is correct.

### Graph contracts

```
encoder  in  mel[1,128,2885] f16            out enc_proj[1,361,640]     (GPU gate cos 0.999995 vs fp32)
predict  in  token[1,1] i32 · h[2,1,640] f32 · c[2,1,640] f32  out dec_out[1,640] · h_out · c_out
joint    in  dec_out[1,640] f32 · enc_frame[1,640] f32          out token_logits[1,8193] · dur_logits[1,5]
```

Constants: blank **8192**, vocab **8193**, durations **[0,1,2,3,4]**, decoder_start 8192, 16 kHz.

## Numerics gate (end-to-end, on-engine + in Swift)

The whole pipeline was gated **token-for-token** vs the HF `ParakeetForTDT` reference on a
LibriSpeech clip: **77/77 exact** —

> "With her white paint and her scarlet smokestack, the Inverashiel, one of the two small steamers
> that during the summer months plied up and down the loch, and incidentally carried on
> communication between Inverashiel and Cryonon."

Re-authored in plain torch from `model.safetensors` (encoder eager cos 1.000000, 710 tensors),
exported, and gated on Core AI **GPU**. Then the **Swift** path (`KitParakeetModel`: Accelerate mel
→ the three graphs → host TDT loop → swift-transformers detokenize) reproduced the same 77 tokens
**token-exact on Mac GPU** — the engine is a CoreAIKit drop-in, no Python at inference.

## The port in three lessons

1. **The joint activation is ReLU, not the encoder's SiLU.** The joint uses the top-level
   `config.hidden_act` (relu); wiring the encoder's silu there silently garbles the transcript.
2. **`ParakeetFeatureExtractor` always per-utterance normalizes.** Despite a `do_normalize` arg
   (dead code in transformers 5.12.1), the extractor *always* zero-means/unit-vars the log-mel per
   channel. The encoder was gated on the normalized features, so the Swift mel must normalize too —
   and the bucket is filled by **silence-padding the audio** (the trailing-silence frames are
   normalized in place), **not** by padding the mel with a constant. Padding the mel with raw zeros
   instead makes the decoder hallucinate extra tokens over the tail.
3. **fp16 encoder, fp32 decoder, all on GPU.** The 24-layer FastConformer ships fp16 (GPU cos
   0.999995; CPU fp16 is noisier, min ~0.95). The small predictor/joint stay fp32.

## ⬇️ Bundle

**`mlboydaisuke/Parakeet-TDT-0.6B-CoreAI`** *(upload pending)* — `encoder` (fp16, 1.2 GB) +
`predict` (fp32, 49 MB) + `joint` (fp32, 21 MB) `.aimodel`s + `tokenizer.json`. cc-by-4.0. CoreAIKit
drop-in: `KitParakeetModel` (`transcribe(samples:onPartial:)`). In the **coreai-audio** app's
Transcribe tab as "Parakeet TDT 0.6B".

Convert yourself: [`conversion/parakeet/`](../conversion/parakeet/) — `gen_oracle.py` (golden, in an
isolated transformers-5.x env) → `export_encoder.py` (FastConformer re-author) → `export_decoder.py`
(predictor + joint) → `gate_e2e.py` (full pipeline) and `gate_mel_swift.py` (the Swift mel recipe,
gated token-exact e2e).
