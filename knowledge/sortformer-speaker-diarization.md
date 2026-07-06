# Streaming speaker diarization: exporting only the neural core, porting the loop to the host

Lessons from porting NVIDIA's [`diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
(NeMo, cc-by-4.0, 117M) to Core AI — streaming **speaker diarization** ("who spoke when", ≤4
speakers). The model is a FastConformer encoder + an 18-layer post-LN Sortformer transformer with a
stateful **streaming** wrapper (a speaker cache that is compressed every chunk). The transferable
move: **export only the stateless core as a fixed-shape graph and re-author the entire streaming
algorithm in the Swift host.** Transferable to any NeMo streaming model (and to Parakeet/Nemotron,
which share the 128-mel frontend). Conversion + the authoritative reference:
[`conversion/sortformer_diar`](../conversion/sortformer_diar).

## 1. Export boundary: the stateless core only

`forward_streaming` carries Python-side state (speaker cache, FIFO, mean-silence embedding, per-chunk
compression) that no static graph can hold. But NeMo already ships a `forward_for_export` that is a
**pure function of fixed-size buffers** — that is the only thing worth tracing:

```
chunk_mel [1,1520,128]  (host zero-pads each mel chunk)
spkcache  [1,188,512]   (host-maintained speaker cache)
valid     [1,378]       (1 = real frame / 0 = pad)
  -> preds    [1,378,4] (sigmoid speaker activity)
     chunk_pe [1,190,512] (pre-encode embeddings; the host folds them back into the cache)
```

The graph outputs **both** `preds` and `chunk_pe` — the host needs the embeddings to update the
speaker cache, exactly as NeMo's export does. Everything stateful (the chunk loop, the cache
compression, the silence profile) is re-authored in Swift. Same idiom as the other host-DSP audio
ports (Kokoro, Stable Audio, Mel-Band RoFormer): **graph = the tensor math, host = the algorithm.**

## 2. Masks from one `valid` vector; the async/188 layout equals the sync concat

The graph derives every attention/conv mask **in-graph** from a single `valid [1,378]` vector — an
additive attention bias plus a multiplicative conv mask (equivalent to NeMo's `masked_fill` because
the depthwise conv re-zeroes padded frames each layer). So the host never builds masks; it just sets
`valid[0:spkcache_len]=1` and `valid[188:188+chunk_pe_len]=1`.

The fixed **188** offset is the *async* streaming convention, but for the valid region it is
numerically identical to the *sync* `concat_embs` path (relative-position shift-invariance + local
conv + masking). Verified byte-exact — so you can ship the simpler fixed-buffer layout regardless of
which streaming mode the config selects.

## 3. Host 128-mel = the FastConformer frontend, `normalize=NA`; watch the log/silence cosine trap

The frontend is NeMo `AudioToMelSpectrogramPreprocessor` (preemph 0.97 → STFT n_fft=512 / win=400
Hann centered / hop=160 → librosa-slaney mel[128,257] → `log(mel + 2⁻²⁴)`), **`normalize=NA`** (no
per-channel normalization — the only delta from Parakeet's mel, which normalizes). Reuse CoreAIKit's
`ParakeetMelPreprocessor` structure and drop the normalization; ship the filterbank as a `.f32`.

**Cosine trap.** The exported host mel scored **cos 0.995 vs the captured golden mel**, which looks
alarming. Per-stage diff showed the STFT power and the **linear** mel are `cos 1.000` (bit-identical
to NeMo); the gap appears *only after the log*. Cause: `log` compresses speech and **expands the
near-zero silence floor** (`log(2⁻²⁴) ≈ -16.8`), so tiny fp differences there — plus NeMo's
training-time `dither=1e-5` on the golden — dominate the cosine even though the speech bins match
exactly. Irrelevant to diarization (the decision is a 0.5 threshold): end-to-end activity agreement
stayed **100%**. Lesson: **gate log-mel on the pre-log linear stage (or on the downstream decision),
not on a whole-spectrogram cosine** that the silence floor skews.

## 4. Re-implement AOSC from the canonical source, not a demo-passing re-impl

The speaker cache is compressed every chunk (AOSC: log-pred scores → disable low/overlap scores →
strong/weak top-k boost → append silence pad rows → top-k across speakers → gather). Porting this
1:1 from NeMo `sortformer_modules.py` (`permute_spk=False`, inference path) is mechanical **but has a
trap**: `_get_topk_indices` takes `n_frames = scores.shape[1]` — the **padded** count (after the
`SIL_PER_SPK` `inf` rows are appended), so `remainder` wraps by `F+SIL` and the silence rows land at
`>= n_frames_no_sil` and get disabled. An intermediate Python re-impl that passed `n_frames` = the
**pre-pad** count instead shifted every real frame index by `SIL/speaker` — invisible on the demo,
wrong on real clips. **Derive the wrap length from the padded tensor, exactly as the source does.**

## 5. A 2-chunk clip never exercises compression — gate a long clip

`compress_spkcache` only runs once the cache exceeds `spkcache_len` (188). On the 21.5 s demo the
cache reaches 270 **after the last chunk**, so compression runs but **never affects the output** —
the demo gate is 100% even with a broken compressor (that is how the §4 bug hid). Capture a longer
reference (here: the demo clip ×3 = 64.5 s → 808 output frames, compression fires ~4×) and gate the
accumulated `total_preds` against NeMo `forward_streaming`. Only then is the cache math validated.
Result: **100% activity agreement** on both clips, in Python, in Swift on Mac GPU, and on iPhone 17
Pro (AOT h18p) — all driving the same exported fp16 graph.

## 6. Product framing: parity, and diarize-then-transcribe (no ASR word timestamps needed)

On-device Sortformer already exists (FluidAudio / soniqo on CoreML/ANE), so this is **speed parity,
not a novelty**. The differentiator is the integration: a **diarized transcript** wired to the zoo's
own ASR. CoreAIKit's `Transcription` exposes only `{language, text}` — **no word timestamps** — so
word-to-frame alignment is impossible. The clean alternative: **diarize into speaker turns, then run
the ASR on each turn's audio slice** → `Speaker N [t0–t1]: text`. The diarizer supplies the
boundaries; no timestamp API needed. Robust and simpler than alignment.

Practical: diarization keys on **speaker embeddings**, so it splits cleanly only when the voices are
acoustically distinct (a male+female interview → clean 2-speaker split; two similar male radio voices
or one dominant speaker → collapses to one). Pick demo audio with distinct speakers.
