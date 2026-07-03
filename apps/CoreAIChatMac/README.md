# CoreAIChatMac

A minimal macOS chat app for **Core AI language bundles on Apple's official runtime**
([apple/coreai-models](https://github.com/apple/coreai-models)). Point it at a folder
of exported bundles, click a model, chat — with live performance stats (load time,
TTFT, tok/s, memory footprint) in the footer.

Runs anything `coreai.llm.export` produces: gpt-oss-20b, qwen3, gemma3, mistral —
and zoo bundles. gpt-oss's harmony output is parsed into a collapsible "Thinking"
section + the final answer.

## Build & run

```bash
brew install xcodegen
cd apps/CoreAIChatMac
xcodegen generate
open CoreAIChatMac.xcodeproj   # Run (scheme uses Release — debug engine is ~3x slower)
```

Export a model first, e.g.:

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
uv run coreai.llm.export openai/gpt-oss-20b
```

Then in the app: **Choose Models Folder…** → select the `exports/` directory.

## Download models in-app

**Download Models…** in the sidebar streams published `.aimodel` bundles straight from
Hugging Face into the app's models directory (resumable chunked transfer —
`AppShared/ModelDownloader.swift`), so no local export is needed. The catalog
(`ModelCatalog.swift`) covers two families:

- **Official-recipe** (stock runtime, `macos/<name>.aimodel` in each `*-CoreAI-official`
  repo): Qwen3 0.6B/4B/8B, Gemma 3 4B/12B IT, Mistral 7B v0.3, gpt-oss 20B.
- **Zoo community ports** (engine patches / custom Metal kernels, `gpu-pipelined/<bundle>`):
  Qwen3.6-35B-A3B, Qwen3.6-27B, GLM-4.7-Flash, LFM2.5-8B-A1B, Gemma 4 12B/31B.

## Notes

- The model list shows any subdirectory containing a `metadata.json`.
- **Prefix caching (cross-turn KV reuse):** each turn keeps the KV cache for the
  longest common prefix with the previous turn and prefills only the new tokens,
  instead of re-processing the whole conversation. Lossless; at a 4k-token context a
  follow-up's time-to-first-token drops ~23 s → 0.2 s. Needs the engine `trimKVCache`
  primitive — apply `apps/coreai-prefix-cache.patch` to the `coreai-models` checkout
  (details in `knowledge/prefix-cache-kv-reuse.md`). Toggle off for A/B with
  `CHATMAC_NO_PREFIX_CACHE=1`.
- Generation: temperature 0.7, max 2048 tokens per reply, official
  `VanillaDecodingStrategy` streaming.
- Measured on M4 Max 128GB: gpt-oss-20b decodes at ~78 tok/s
  (see `knowledge/apple-models-bench.md` for the full benchmark table).
