# Whisper ASR on the stock runtime: the fixed-window autoregressive decode

Lessons from porting [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
to Core AI via Apple's **official** `models/whisper/export.py` recipe, and making it actually
**transcribe** on the **stock** runtime (no engine patch). Transferable to any encoder-decoder
seq2seq model exported as a single combined graph.

## 1. The official recipe traces a single decode step — it can't transcribe

`models/whisper/export.py` traces the model with `decoder_input_ids` of shape **`[1, 1]`** and
**no KV-cache state** (`state_names == []`). The exported graph is:

```
input_features [1, 128, 3000] (log-mel) + decoder_input_ids [1, 1] → logits [1, 1, 51866]
```

That is a *single-position* forward. Driven autoregressively by re-feeding only the last token,
every step is "position 0" with no prior context (self-attention sees one token, the prompt is
gone), so it emits nothing useful — measured: empty transcript. The recipe is a conversion
demonstration, not a runnable ASR pipeline. **A combined encoder-decoder graph needs either a
growing decoder input or a KV cache to be usable; this one has neither.**

## 2. Dynamic decoder length is correct but recompiles every step (~15 s/token)

The obvious fix: re-export with the decoder sequence dimension dynamic
(`torch.export` `dynamic_shapes={"decoder_input_ids": {1: Dim("dec", 1, 448)}}` → descriptor
`[1, -1]`), then feed the **whole growing prefix** each step and read the last position. This is
**token-for-token identical** to the HF PyTorch greedy reference — but each step has a *different*
concrete length, so MPSGraph **recompiles the whole graph per length**: measured **~15 s/token**
(164 s for 11 tokens). Same class of trap as the [growing-shape decode fault](unlimited-ocr-rswa-static-decode.md) —
a moving shape is death.

## 3. The fix: a FIXED decoder window + read at the real last position

Re-export with `decoder_input_ids` traced at a **fixed length** (128 here, ≤ Whisper's 448 max):

```python
"decoder_input_ids": torch.zeros((1, 128), dtype=torch.int32)   # fixed, no dynamic_shapes
```

At decode time, put the real tokens at `[0..k]`, pad the rest of the 128-slot buffer (any value —
EOT is fine), run, and read **`argmax(logits[0, k])`** at the real last index `k`:

- **Causal self-attention** means position `k` never attends to positions `> k`, so the padding
  after the real tokens **cannot affect** the logits at `k` — the read is exact.
- The shape is **constant** every step → MPSGraph compiles **once**.

Result: token-for-token identical to PyTorch greedy, **first step 0.68 s (compile+warmup), then
0.18 s/token** on M4 Max GPU. The combined graph re-encodes the audio every step (wasteful — the
encoder is the heavy part of "turbo"), but at a constant shape it's pure compute, not recompile,
and 0.18 s/token is fine for a 30 s window. (A further win, not needed here, is splitting
encoder/decoder so the encoder runs once — a bigger deviation from the recipe.)

The fixed window caps a single decode at `128 − len(prompt)` tokens (enough for one 30 s window;
chunk longer audio). Recipe: [`conversion/export_whisper_fixed.py`](../conversion/export_whisper_fixed.py).

## 4. Compute-unit routing — single `main` graph → GPU, not ANE

Loading the `.aimodel` with raw default options routes to the **ANE and crashes**
(`createProgramInstanceForModel … Program load failure (0x10004)` → `could not load module from
MPSGraphPackage`). The Swift `PreparedModel.prepare(at:)` auto-detects structure: a bundle whose
only graph is `main` is classified `.dynamic` → `SpecializationOptions(preferredComputeUnitKind:
.gpu)`, which is correct. In Python pass it explicitly:
`SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())`.

## 5. The decode prompt and stop token (Whisper large-v3)

```
prompt = [50258, 50259, 50360, 50364]   # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
stop   = 50257                          # <|endoftext|>
```

Force the language token for a known language; omit/auto-detect is not possible with a fixed prompt.
Detokenize the generated ids (after the prompt, before EOT) with the HF tokenizer
(swift-transformers `AutoTokenizer.from(modelFolder:)` → `tokenizer.decode(tokens:)`).

## 6. The log-mel frontend in Swift (n_fft = 400 isn't FFT-friendly)

Whisper's feature extractor is `n_fft=400, hop=160, 128 mels, 16 kHz, 30 s → [1,128,3000]`. **400 is
not a vDSP-FFT length** (vDSP_DFT wants `2^n · {1,3,5,15}`; 400 = 16·25). Zero-padding to 512 and
taking 201 bins is wrong — the bins land at different frequencies than the 400-pt FFT the mel
filterbank was designed for.

Compute the 400-point DFT as a **matmul against a precomputed cos/sin basis** `[400 × 201]`
(`vDSP_mmul`), then the mel filterbank (shipped as `mel_filters_128.npy`, `[201 × 128]`) as a second
matmul, then `log10` → clamp to `max − 8` → `(x + 4)/4`. Match torch.stft: reflect-pad `n_fft/2` on
each side (center=True), Hann window, drop the last frame, pad/trim to 3000. See
[`apps/CoreAITranscribe/Sources/WhisperMel.swift`](../apps/CoreAITranscribe/Sources/WhisperMel.swift).
Ship the mel filterbank with the bundle so the app doesn't recompute it.

## 7. Verify against the PyTorch reference — and watch the test audio

Gate the Core AI greedy transcript against `model.generate(..., do_sample=False, num_beams=1)`; they
should be **token-for-token identical** (greedy is deterministic). Gotcha: macOS `say` uses the
**system-locale** voice — on a Japanese system it produced Japanese-accented English that Whisper
auto-detected as Japanese (katakana transcript). Use `say -v Alex` and force `language="en"`.

float16 ships (~1.5 GB, iOS-friendly); it matched the float32 reference here.
