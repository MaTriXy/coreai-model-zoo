# GLiNER2-PII → Core AI — porting notes

The zoo's first NER / schema-driven extraction model and first DeBERTa-v3 port. Base:
`fastino/gliner2-privacy-filter-PII-multi` (Apache-2.0) on `microsoft/mdeberta-v3-base` (278M, 251k
multilingual vocab). The whole model is one static Core AI graph; the tokenizer, schema
linearization, and span decode are a byte-exact Swift port of GLiNER2's Python inference path. Four
lessons were worth the session.

## 1. DeBERTa-v3 disentangled attention exports cleanly at a fixed shape — no rewrite

The one feared gate was DeBERTa-v3's disentangled attention (content→position + position→content
terms over relative-position *buckets*). It exports with **no hand-written rewrite**: run the stock
`DebertaV2Model` at a fixed `[1, 256]` shape through `torch.export` → `run_decompositions` →
`TorchConverter.to_coreai()`. `coreai_torch`'s decomposition handles the bucket gathers as long as
the shape is static, and the GPU engine matches eager per-token (cos min 0.99962 / mean 0.99992). A
fixed shape also sidesteps the OS-27 dynamic-query JIT wedge. Fixed shape is the whole trick.

## 2. The fused graph is schema-agnostic — labels are host input, not baked weights

GLiNER's value is zero-shot: any label set at call time. That survives conversion because the label
set enters through `input_ids`, not the weights. The graph is:

```
forward(input_ids[1,256], attention_mask[1,256], text_word_idx[1,96], schema_idx[1,17])
    -> span_scores[1,16,96,8]         # [MMAX, T, K]
```

mDeBERTa-v3 → `index_select` "first" word pooling → SpanMarker → CountLSTM(count=1) → einsum →
sigmoid. The host linearizes the user's labels into the token stream
(`( [P] entities ( [E] l0 [E] l1 … ) ) [SEP_TEXT] <text words>`) and passes the gather positions
(`text_word_idx` = each word's first sub-word; `schema_idx` = the `[P]` + per-label `[E]` marker
positions, padded to `MMAX+1` with the `[P]` position). `count` is fixed to 1 because entity
extraction reads only count-step 0. One converted bundle answers any schema up to `MMAX=16` — verified
by running label sets never seen at export (credentials, org/money/date/location) and matching
`ext.extract` byte-for-byte.

## 3. swift-transformers tokenizer routing — the make-or-break

The Swift host must reproduce mDeBERTa's tokenization **byte-for-byte** or the graph's `input_ids`
are wrong. Two non-obvious facts, both verified against Python fixtures:

- **`DebertaV2Tokenizer` → alias to `XLMRobertaTokenizer`.** swift-transformers (1.3.3) has no
  DebertaV2 class, and its *non-strict fallback is BPE* — silently wrong for what is really an
  SPM/Unigram tokenizer. mDeBERTa-v3 is the same SentencePiece/Unigram family as XLM-R, so rewriting
  the exported `tokenizer_config.json`'s `tokenizer_class` to `XLMRobertaTokenizer` routes it through
  the correct `UnigramTokenizer`. Per-word `tokenize()` then matches HF exactly for URLs, emails,
  accented text (café/naïve), hyphenated tokens, and numbers.
- **GLiNER special markers live above the Unigram vocab.** `[P] [E] [SEP_TEXT] …` are `added_tokens`
  at ids 250102+, past the 250101-entry Unigram vocab, so swift's `convertTokensToIds` returns
  `[UNK]` for them. The collator emits their ids directly from a small map (`extractor.json`).

GLiNER calls `tokenizer.tokenize(word)` **per word** (not full-sequence `encode`), so no CLS/SEP is
added; the collator concatenates per-word sub-words and converts to ids itself. The exported bundle
carries `extractor.json` (graph shapes S/T/MMAX/K, threshold, the `entities` prompt token, and the
marker/pool ids) so the kit needs nothing hard-coded.

## 4. Host collator + decode are byte-exact ports of GLiNER2

The Swift `InformationExtractor` reproduces `SchemaTransformer.collate_fn_inference` (word-split
regex → schema linearization → per-word tokenize → gather indices) and `_format_spans` (per-label
threshold + confidence-descending greedy NMS over character spans). Both are byte-identical to Python
on the demo PII suite, so the only fp16 slack is the graph itself (span-scores cos 0.999993), which
does not flip any decoded entity.

## iOS

Fixed-shape graph → clean AOT: `coreai-build compile <aimodel> --output <dir> --platform iOS
--architecture h18p --preferred-compute gpu` (no `--expect-frequent-reshapes` needed). The 823 MB
h18p `.aimodelc` loads on iPhone 17 Pro in ~1.8 s (device JIT skipped), and extraction is ~22–32 ms
per text warm. Rename the `.aimodelc` to `.aimodel` where the kit scans for it (`AIModel` auto-detects
AOT). The HF repo ships `macos/` (JIT) and `ios/` (AOT) subtrees; `ModelID.gliner2PII` has `path: nil`
so `resolvedPath` picks the right one.

## Reproduce

```bash
python conversion/export_gliner2_pii.py --output-dir /tmp/gliner2_out --dtype float16
#   GATE-1 fp32 span-scores cos 1.0 · GATE-2 decoded == ext.extract · GATE-3 engine cos 0.999993
# coreai-kit:
swift run infoextract-cli --bundle /tmp/gliner2_out --gate            # kit E2E, ALL PASS
swift run infoextract-cli --bundle /tmp/gliner2_out --redact --text "email me at a@b.com"
```
