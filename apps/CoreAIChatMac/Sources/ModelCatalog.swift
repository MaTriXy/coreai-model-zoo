// ModelCatalog — the large, Mac-only Core AI bundles published on Hugging Face,
// downloadable in-app. Each `remote` is the bundle directory inside its repo (it
// holds metadata.json + the .aimodel + tokenizer), `local` is the folder created
// under the app's models directory — ChatEngine.scanFolder then picks it up.
import Foundation

struct DownloadableModel: Identifiable, Hashable {
    let id: String
    let name: String          // display name
    let detail: String        // short architecture note
    let repo: String          // Hugging Face repo URL
    let remote: String        // bundle directory path inside the repo
    let approxSizeGB: Int
    var localName: String? = nil  // explicit local dir name; else the remote's last component

    // The official repos hold the LanguageBundle at `macos/` (same name in every repo), so
    // those entries pass an explicit per-model `localName` to avoid colliding under models/.
    var local: String { localName ?? (remote as NSString).lastPathComponent }
    var item: ModelDownloader.Item { .init(remote: remote, local: local) }
}

enum ModelCatalog {
    // Two families, all loaded by ChatEngine on the same `coreai-sequential` path:
    //   • Official-recipe bundles (Apple's `coreai.llm.export`, unmodified) — stock
    //     runtime, hosted at `macos/<name>.aimodel` in each *-CoreAI-official repo.
    //     They are plain `_dynamic` LanguageBundles (coreai-sequential drives any
    //     dynamic bundle); gpt-oss harmony output is split by HarmonyParser. tok/s =
    //     M4 Max decode from knowledge/apple-models-bench.md.
    //   • Zoo community ports — engine patches / custom Metal kernels, all under
    //     `gpu-pipelined/<bundle>` (HF tree verified 2026-06-15).
    static let macModels: [DownloadableModel] = [
        // ── Official-recipe (stock runtime). The LanguageBundle IS the repo's `macos/`
        //    dir (metadata.json 0.2 + the inner .aimodel + tokenizer/), so `remote` is
        //    "macos" and an explicit `localName` keeps the bundles from colliding. ──
        DownloadableModel(
            id: "qwen3-0.6b", name: "Qwen3 0.6B", detail: "official · dense 4-bit · 484 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/qwen3-0.6b-CoreAI-official",
            remote: "macos", approxSizeGB: 1, localName: "qwen3_0_6b_official"),
        DownloadableModel(
            id: "qwen3-4b", name: "Qwen3 4B", detail: "official · dense 4-bit · 145 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/qwen3-4b-CoreAI-official",
            remote: "macos", approxSizeGB: 2, localName: "qwen3_4b_official"),
        DownloadableModel(
            id: "gemma3-4b", name: "Gemma 3 4B IT", detail: "official · dense 4-bit · 142 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/gemma-3-4b-it-CoreAI-official",
            remote: "macos", approxSizeGB: 2, localName: "gemma_3_4b_it_official"),
        DownloadableModel(
            id: "mistral-7b", name: "Mistral 7B v0.3", detail: "official · dense 4-bit · 102 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/mistral-7b-v0.3-CoreAI-official",
            remote: "macos", approxSizeGB: 4, localName: "mistral_7b_v0_3_official"),
        DownloadableModel(
            id: "qwen3-8b", name: "Qwen3 8B", detail: "official · dense 4-bit · 94 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/qwen3-8b-CoreAI-official",
            remote: "macos", approxSizeGB: 4, localName: "qwen3_8b_official"),
        DownloadableModel(
            id: "gemma3-12b", name: "Gemma 3 12B IT", detail: "official · dense 4-bit · 55 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/gemma-3-12b-it-CoreAI-official",
            remote: "macos", approxSizeGB: 6, localName: "gemma_3_12b_it_official"),
        DownloadableModel(
            id: "gpt-oss-20b", name: "gpt-oss 20B", detail: "official · MoE MXFP4 · 78 tok/s · ~34 GB RAM",
            repo: "https://huggingface.co/mlboydaisuke/gpt-oss-20b-CoreAI-official",
            remote: "macos", approxSizeGB: 13, localName: "gpt_oss_20b_official"),
        // ── Zoo community ports (engine patches / custom Metal kernels) ──
        // Diffusion LM (zoo's first): masked-diffusion decode via a host loop over a static
        // bidirectional forward (LLaDAEngine), NOT the AR runtime. int4 body + int8 head ~4.9 GB.
        DownloadableModel(
            id: "llada-8b-dllm", name: "LLaDA-8B (diffusion)", detail: "dLLM · d3LLM distilled · parallel denoising · int4",
            repo: "https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI",
            remote: "macos", approxSizeGB: 5, localName: "llada8b_dllm"),
        DownloadableModel(
            id: "qwen36-35b", name: "Qwen3.6-35B-A3B", detail: "MoE · 35B/~3B active · gather_qmm",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI",
            remote: "gpu-pipelined/qwen3_6_35b_a3b_decode_sym8_gather", approxSizeGB: 35),
        DownloadableModel(
            id: "qwen36-27b", name: "Qwen3.6-27B", detail: "dense · int8 == fp16 quality",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_6_27b_decode_int8hu_block32_sym", approxSizeGB: 28),
        // ⚡Spec pair: the 27B verify graph (static S=9 window) + its 0.8B int4 draft.
        // Lossless speculative decoding via SpecDecodeEngine — download BOTH; selecting
        // the 27B ⚡Spec bundle auto-pairs the draft from the same models folder.
        DownloadableModel(
            id: "qwen36-27b-spec", name: "Qwen3.6-27B ⚡Spec", detail: "dense · lossless spec-decode · needs the 0.8B draft",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_6_27b_verify_s9_int8hu_block32_sym", approxSizeGB: 28),
        DownloadableModel(
            id: "qwen35-08b-draft", name: "Qwen3.5-0.8B (⚡Spec draft)", detail: "int4 draft for Qwen3.6-27B ⚡Spec",
            repo: "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI",
            remote: "gpu-pipelined/qwen3_5_0_8b_verify_s9_int4lin", approxSizeGB: 1),
        DownloadableModel(
            id: "qwen38-27b", name: "Qwen3.8-27B", detail: "dense · int8",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_8_27b_decode_int8hu_block32_sym", approxSizeGB: 28),
        DownloadableModel(
            id: "qwen38-27b-int4", name: "Qwen3.8-27B (int4)", detail: "dense · int4 · 22 tok/s — the free-form pick",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_8_27b_decode_int4lin", approxSizeGB: 18),
        // 3.8 ⚡Spec set: S=9 d4 hpost int4 verify graph (--emit-hidden post) + the
        // checkpoint's own MTP head as drafter (S=1) with the S=9 replay sibling;
        // metadata `spec_draft`/`spec_mtp_replay` auto-pair them (the 0.8B d4 draft
        // stays a SPEC_DRAFT= A/B option). Code 29.0 tok/s lossless; for free-form
        // prose use the plain int4 decode bundle (spec loses at c_v 2.43).
        DownloadableModel(
            id: "qwen38-27b-spec", name: "Qwen3.8-27B ⚡Spec", detail: "int4 · lossless spec-decode · needs the MTP drafter pair",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_8_27b_verify_s9_int4lin_d4_hpost", approxSizeGB: 18),
        DownloadableModel(
            id: "qwen38-27b-mtp", name: "Qwen3.8-27B (⚡Spec MTP drafter)", detail: "the checkpoint's own 1-layer MTP head · int8",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_8_27b_mtp_s1_int8hu_block32_sym", approxSizeGB: 5),
        DownloadableModel(
            id: "qwen38-27b-mtp-replay", name: "Qwen3.8-27B (⚡Spec replay)", detail: "S=9 batched-replay sibling · +33% on code",
            repo: "https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI",
            remote: "gpu-pipelined/qwen3_8_27b_mtp_s9_int8hu_block32_sym", approxSizeGB: 5),
        DownloadableModel(
            id: "qwen35-08b-draft-d4", name: "Qwen3.5-0.8B (⚡Spec d4 draft)", detail: "int4 draft for Qwen3.8-27B ⚡Spec",
            repo: "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI",
            remote: "gpu-pipelined/qwen3_5_0_8b_verify_s9_int4lin_d4", approxSizeGB: 1),
        DownloadableModel(
            id: "glm47", name: "GLM-4.7-Flash", detail: "MoE + MLA · 30B/~3B active",
            repo: "https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI",
            remote: "gpu-pipelined/glm_4_7_flash_decode_sym8_gather", approxSizeGB: 30),
        DownloadableModel(
            id: "lfm2-8b-a1b", name: "LFM2.5-8B-A1B", detail: "MoE · conv+attn hybrid · ~1.5B active · gather_qmm",
            repo: "https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI",
            remote: "gpu-pipelined/lfm2_5_8b_a1b_decode_sym8_gather", approxSizeGB: 9),
        DownloadableModel(
            id: "gemma4-12b", name: "Gemma 4 12B", detail: "dense · int8 · flash-decode kernel",
            repo: "https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI",
            remote: "gpu-pipelined/gemma4_12b_qat_decode_int8lin_msdpa_g8", approxSizeGB: 13),
        DownloadableModel(
            id: "gemma4-31b", name: "Gemma 4 31B", detail: "dense · int4 QAT · flash-decode kernel",
            repo: "https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI",
            remote: "gpu-pipelined/gemma4_31b_qat_decode_int4linsym_msdpa_g8", approxSizeGB: 18),
        // Agentic coder (DeepReinforce, Qwen3.5 hybrid arch). int8 gate: 24/24 exact vs
        // fp32 oracle; 48.3 tok/s on M4 Max. int4lin sibling (58.9) also on the repo.
        DownloadableModel(
            id: "ornith-9b", name: "Ornith-1.0-9B", detail: "agentic coder · dense int8 · 48 tok/s",
            repo: "https://huggingface.co/mlboydaisuke/Ornith-1.0-9B-CoreAI",
            remote: "gpu-pipelined/ornith_1_0_9b_decode_int8hu_block32_sym", approxSizeGB: 10),
    ]
}

extension ChatEngine {
    // App-managed download location (macOS-appropriate): ~/Library/Application Support/CoreAIChatMac/models
    static var appModelsDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreAIChatMac/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
}
