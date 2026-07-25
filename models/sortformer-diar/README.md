# Streaming Sortformer 4-spk v2 — Core AI

The zoo's **first speaker-diarization** model — "who spoke when", up to 4 speakers, streaming and
fully on-device. [`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
(cc-by-4.0, 117M) with only its **neural core** exported as a Core AI graph; the NeMo 128-mel
frontend, the streaming chunk loop, and the **AOSC speaker-cache compression** run in the Swift host.

> ⚠️ **License**: use the **streaming v2** checkpoint (CC-BY-4.0). The offline `diar_sortformer_4spk-v1`
> is CC-BY-**NC** and is not used here.

> **Parity, not novelty.** On-device Sortformer already exists (e.g. FluidAudio / soniqo on
> CoreML/ANE). This is speed parity — shipped as a **diarized transcript** wired to the zoo's own ASR,
> not as a diarization first.

## Use it

Ships in the **[coreai-audio](../../apps/coreai-audio)** app, Transcribe tab → **"Diarize — who said
what"**. The diarizer segments each speaker turn; the on-device ASR you picked (Whisper / Qwen3-ASR /
Parakeet / Nemotron) transcribes each turn into a **diarized transcript**:

```
Speaker 1 [0.3–4.1s]: …
Speaker 2 [4.6–6.3s]: …
```

No ASR word timestamps are needed — the diarizer supplies the turn boundaries.

## How it works

The exported graph is the static `forward_for_export` core (masks derived in-graph from one `valid`
vector); everything else is a 1:1 Swift port of NeMo `sortformer_modules.py` (inference path):

- **Fixed-buffer graph** — `chunk_mel[1,1520,128]` + `spkcache[1,188,512]` + `valid[1,378]` →
  `preds[1,378,4]` (sigmoid activity) + `chunk_pe[1,190,512]` (embeddings the host folds back into the
  speaker cache). `pos_emb` baked as a constant.
- **Host frontend** — NeMo 128-mel (preemph 0.97 → STFT n_fft=512 / win=400 / hop=160 → librosa-slaney
  mel → log, `normalize=NA`). STFT + mel are bit-identical to NeMo (cos 1.0 pre-log).
- **Host streaming loop** — chunk the mel (188·8 frames, ±1 subsample ctx), run the graph per chunk,
  slice the chunk preds, `streaming_update` + `compress_spkcache` (AOSC: log-pred scores → disable
  low → strong/weak top-k boost → silence pad → top-k / gather), carry the speaker cache across
  chunks. Threshold 0.5 per frame per speaker (frame = 80 ms) → speaker turns.

## Verification

Byte-gated vs NeMo `forward_streaming` at **100.00 % speaker-activity agreement** (@0.5) on a 21.5 s
and a 64.5 s clip (the latter exercises the AOSC cache compression ~4×) — in Python, in **Swift on Mac
GPU**, and on **iPhone 17 Pro** (A19 Pro, AOT h18p), all driving the exported fp16 graph. Reproduce:

```bash
coreai-models/.venv/bin/python conversion/sortformer_diar/gate_e2e_engine.py   # shipped .aimodel, demo, 100%
coreai-models/.venv/bin/python conversion/sortformer_diar/gate_long.py         # 64.5s (compress ~4x), 100%
```

Conversion + the authoritative Swift-port reference:
[`conversion/sortformer_diar`](../../conversion/sortformer_diar) (`HANDOFF.md`). Porting lessons (export
boundary, the log-mel cosine trap, the AOSC compression bug, the long-clip gate):
[`knowledge/sortformer-speaker-diarization.md`](../../knowledge/sortformer-speaker-diarization.md).

- 🤗 [Streaming-Sortformer-Diar-CoreAI](https://huggingface.co/mlboydaisuke/Streaming-Sortformer-Diar-CoreAI)
  — macOS `.aimodel` + iOS `.h18p.aimodelc` + mel filterbank.
- Base model: [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) (CC-BY-4.0).
