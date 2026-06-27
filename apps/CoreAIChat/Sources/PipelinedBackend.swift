// PipelinedBackend — decode-only loop-free int8lin LanguageBundles on Apple's
// coreai-pipelined GPU engine: async non-blocking encode, on-GPU argmax
// sampling, on-device KV growth, zero custom kernels. One class drives every
// pipelined chat model; `Spec` picks the bundle (downloaded from HF into
// Documents/models/):
//   * Qwen3.5-0.8B int8hu — 69.7–74.0 tok/s decode on iPhone 17 Pro (absmax
//     int8 head; the HF dir is named perchan_sym but the content is
//     per-block-32 — vs 50.3–51.5 for int8lin, 42.5–45.4 for the kernel
//     monolith)
//   * LFM2.5-1.2B int8lin  — 38.0–39.6 tok/s decode (~87% of the naive BW
//     ceiling; conv+attention hybrid, first non-Qwen rider)
//   * Qwen3.5-2B int8lin   — 19–21 tok/s decode; the 2.4 GB bundle needs the
//     increased-memory entitlement (this app has it) and pays a ~30 s cold
//     GPU specialization on first load
//   * Granite-4.0-H-1B int8lin — Mamba2+attention hybrid (first SSM-scan rider)
//   * Gemma 4 E2B int4lin tbl  — 30.3 tok/s decode / 38.9 prefill settled (vs
//     22–24 for the kernel monolith): the 2.35 GB per-layer-embedding table
//     rides as a STATIC graph input (in-graph gather), bound once as owned
//     MTLBuffers via EngineOptions.staticInputBuffers; the bundle is AOT
//     h18p (.aimodelc) because on-device specialization dies on this graph.
// Requires the engine patch stack on the local coreai-models package
// (extra states / per-token inputs / static inputs — see
// coreai-models-community/apps/README.md).
//
// Engine contract for these bundles (input_ids is STATIC [1,1]):
//   * COREAI_CHUNK_THRESHOLD=1 before engine creation — prefill must run as
//     pipelined S=1 steps (prompt tok/s ≈ decode tok/s).
//   * never call engine.warmup() — it warms query length 256, which the S=1
//     graph rejects; a 1-token generate after load is the warmup.

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Metal
import Tokenizers

@MainActor
final class PipelinedBackend {
    struct Spec: Sendable {
        let bundleName: String     // dir under Documents/models/
        let hfRemotePath: String   // subpath inside the mode's HF repo
        let label: String          // stats / status line prefix
        let warmupToken: Int32     // any valid id for the 1-token warmup
        // Manual chat-template fallback (String(format:) with the prompt as
        // %@) used only when the bundle tokenizer's own template isn't picked
        // up by swift-transformers. ChatML for the qwen/lfm families; granite
        // speaks <|start_of_role|>; gemma needs an explicit <bos> (its
        // tokenizer post-processor doesn't add one).
        var fallbackTemplate: String =
            "<|im_start|>user\n%@<|im_end|>\n<|im_start|>assistant\n"
        // Static graph inputs (gemma tbl): graph input name -> file inside
        // `staticInputDir`, a sibling of the bundle under Documents/models/.
        // Each file is read ONCE into an owned storageModeShared MTLBuffer and
        // handed to the engine via EngineOptions(staticInputBuffers:) — bound
        // unchanged on every encode. Owned beats mmap here: file-backed
        // no-copy buffers pay a per-encode residency tax on iOS (~6-7 ms/GB)
        // and a read-only mapping costs ~65 ms/GB/encode on macOS. Owned
        // bytes are dirty memory against the jetsam limit — this app ships
        // the increased-memory-limit entitlement.
        var staticInputDir: String? = nil
        var staticInputFiles: [String: String]? = nil
        // Stop ids beyond the tokenizer's eos (gemma ends turns with
        // <end_of_turn> 106 while its tokenizer's eos is <eos> 1).
        var stopTokens: Set<Int> = []
    }

    // nonisolated: referenced from the (nonisolated) GemmaMode enum.
    // v2 ship bundle (absmax int8 head). An older int8lin dir may still sit in
    // Documents/models — it is simply no longer referenced, not deleted.
    nonisolated static let qwen = Spec(
        bundleName: "qwen3_5_0_8b_decode_int8hu_perchan_sym",
        hfRemotePath: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym",
        label: "Qwen ⚡pipelined",
        warmupToken: 9707)
    nonisolated static let qwen2b = Spec(
        bundleName: "qwen3_5_2b_decode_int8lin",
        hfRemotePath: "gpu-pipelined/qwen3_5_2b_decode_int8lin",
        label: "Qwen 2B ⚡pipelined",
        warmupToken: 9707)
    nonisolated static let lfm2 = Spec(
        bundleName: "lfm2_5_1_2b_instruct_decode_int8lin",
        hfRemotePath: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8lin",
        label: "LFM2.5 ⚡pipelined",
        warmupToken: 1098)
    nonisolated static let granite = Spec(
        bundleName: "granite_4_0_h_1b_decode_int8lin",
        hfRemotePath: "gpu-pipelined/granite_4_0_h_1b_decode_int8lin",
        label: "Granite ⚡pipelined",
        warmupToken: 5000,
        fallbackTemplate: "<|start_of_role|>user<|end_of_role|>%@<|end_of_text|>\n"
            + "<|start_of_role|>assistant<|end_of_role|>")
    // AOT h18p .aimodelc (iPhone 17 Pro class) — the 2.0 GB-constants graph
    // crashes the on-device specializer; the PLE table set is the one the
    // gemma GPU/ANE modes already download (gemma4_gather_raw).
    nonisolated static let gemmaTbl = Spec(
        bundleName: "gemma4_e2b_decode_int4lin_tbl_aotc",
        hfRemotePath: "gpu-pipelined/gemma4_e2b_decode_int4lin_tbl_aotc_h18p",
        label: "Gemma ⚡pipelined",
        warmupToken: 2364,
        fallbackTemplate: "<bos><start_of_turn>user\n%@<end_of_turn>\n<start_of_turn>model\n",
        staticInputDir: "gemma4_gather_raw",
        staticInputFiles: ["ple_table": "embed_per_layer.i8",
                           "ple_scale": "embed_per_layer.scale.f32"],
        stopTokens: [106])

    // MiniCPM5-1B (OpenBMB, Apache-2.0) — a plain LlamaForCausalLM exported with
    // the stock `coreai.llm.export` (FM-format, NOT a gpu-pipelined static-decode
    // bundle). It still rides this pipelined engine: verified the FM bundle runs
    // under `coreai-pipelined` + COREAI_CHUNK_THRESHOLD=1 (Paris, 218 tok/s). Chat
    // eos is <|im_end|> (130073); the ChatML default fallback matches its template.
    nonisolated static let minicpm5 = Spec(
        bundleName: "minicpm5_1b",
        hfRemotePath: "int8",
        label: "MiniCPM5 ⚡pipelined",
        warmupToken: 5000,
        stopTokens: [130073])

    // FastContext-1.0-4B-SFT (Microsoft, MIT) — Qwen3-4B-arch long-context repo
    // exploration agent. Stock coreai.llm.export GPU bundle riding the pipelined
    // engine (same as minicpm5). ChatML default fallback; Qwen3 eos <|im_end|> = 151645.
    nonisolated static let fastcontext = Spec(
        bundleName: "fastcontext_4b_gpu",
        hfRemotePath: "gpu",
        label: "FastContext-4B ⚡pipelined",
        warmupToken: 9707,
        stopTokens: [151645])

    let spec: Spec
    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    // Owned static-table buffers, kept alive for the engine's lifetime.
    private var tableBuffers: [String: StaticInputBuffer] = [:]
    private(set) var ctx = 4096

    init(spec: Spec) { self.spec = spec }

    var loaded: Bool { engine != nil }

    struct GenStats {
        var label = ""
        var prefillTok = 0
        var prefillSec = 0.0
        var decodeTok = 0
        var decodeSec = 0.0
        var summary: String {
            String(format: "%@ · prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s",
                   label,
                   prefillTok, Double(prefillTok) / max(prefillSec, 1e-6),
                   decodeTok, Double(max(0, decodeTok - 1)) / max(decodeSec, 1e-6))
        }
    }

    func load() async throws {
        // S=1 prefill; set before the engine reads ModelConfig.chunkThreshold.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let models = docs.appendingPathComponent("models")
        let dir = models.appendingPathComponent(spec.bundleName)
        let bundle = try LanguageBundle(at: dir)
        ctx = bundle.maxContextLength

        if let files = spec.staticInputFiles, let dirName = spec.staticInputDir {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw Self.err("no Metal device")
            }
            let tableDir = models.appendingPathComponent(dirName)
            var buffers: [String: StaticInputBuffer] = [:]
            for (input, file) in files {
                let url = tableDir.appendingPathComponent(file)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw Self.err("missing static-input table \(dirName)/\(file)")
                }
                buffers[input] = StaticInputBuffer(try Self.ownedBuffer(url: url, device: device))
            }
            tableBuffers = buffers
            let total = buffers.values.reduce(0) { $0 + $1.buffer.length }
            print(String(format: "[pipelined] static tables %@ (%.2f GB owned)",
                         buffers.keys.sorted().joined(separator: ", "), Double(total) / 1e9))
        }

        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
            options: EngineOptions(staticInputBuffers: tableBuffers)
        )
        self.engine = engine
        self.tokenizer = try await bundle.loadTokenizer()

        // queryLength=1 warmup: one throwaway token through the real graph.
        _ = try await run(ids: [spec.warmupToken], maxTokens: 1, eos: nil, onText: { _ in })
        print(Self.memLine("\(spec.label) loaded"))
    }

    func unload() {
        engine = nil
        tokenizer = nil
        tableBuffers = [:]
    }

    // Chat-templated greedy generation, streaming decoded text via onUpdate.
    func generate(_ prompt: String, maxNew: Int, onUpdate: (String) -> Void) async throws -> GenStats {
        guard let tokenizer else { throw Self.err("backend not loaded") }
        let ids = try templatedIds(prompt, tokenizer: tokenizer)
        guard ids.count < ctx - 1 else { throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(ctx)") }
        let budget = min(maxNew, ctx - ids.count - 1)
        let stats = try await run(ids: ids, maxTokens: budget, eos: tokenizer.eosTokenId) { gen in
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
        print(Self.memLine("\(spec.label) gen"))
        return stats
    }

    private func run(
        ids: [Int32], maxTokens: Int, eos: Int?, onText: ([Int]) -> Void
    ) async throws -> GenStats {
        guard let engine else { throw Self.err("backend not loaded") }
        try await engine.reset()
        let stream = try engine.generate(
            with: ids,
            samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens)
        )
        var stats = GenStats(label: spec.label, prefillTok: ids.count)
        var gen: [Int] = []
        let t0 = SuspendingClock.now
        var tGen = t0
        // Throttle the live UI refresh to ~25 fps and EXCLUDE the UI-callback time
        // from the decode timer. Re-decoding the whole token list + pushing a SwiftUI
        // view update on every token is O(n²) and serializes against the model loop,
        // dragging both the measured and experienced rate below the model's true
        // decode speed (worse the longer the output). After this, in-app decode ≈
        // PipelinedBench. (Incremental per-token decode would be O(1) but risks `�`
        // mid-character glitches for CJK, so throttle is the safe pick.)
        var uiSec = 0.0
        var lastEmit = t0
        for try await step in stream {
            if stats.prefillSec == 0 {
                stats.prefillSec = Self.seconds(since: t0)
                tGen = SuspendingClock.now
                lastEmit = tGen
            }
            let tok = Int(step.tokenId)
            stats.decodeTok += 1
            if let eos, tok == eos { break }
            if spec.stopTokens.contains(tok) { break }
            gen.append(tok)
            let now = SuspendingClock.now
            if now - lastEmit >= .milliseconds(40) {
                onText(gen)
                uiSec += Self.seconds(since: now)
                lastEmit = SuspendingClock.now
            }
        }
        let tFlush = SuspendingClock.now   // final flush (the throttle may skip the tail)
        onText(gen)
        uiSec += Self.seconds(since: tFlush)
        stats.decodeSec = Self.seconds(since: tGen) - uiSec
        return stats
    }

    // Chat template via the bundle tokenizer; the spec's manual fallback if
    // the template file isn't picked up by swift-transformers (ChatML for the
    // qwen/lfm families, role markers for granite — bos rides on each
    // tokenizer's own post-processor).
    private func templatedIds(_ prompt: String, tokenizer: Tokenizer) throws -> [Int32] {
        if let ids = try? tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]]) {
            return ids.map { Int32($0) }
        }
        let text = String(format: spec.fallbackTemplate, prompt)
        return tokenizer.encode(text: text).map { Int32($0) }
    }

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let d = SuspendingClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    // Whole file into an OWNED storageModeShared buffer (see Spec.staticInputDir
    // for why owned beats mmap on both platforms).
    private static func ownedBuffer(url: URL, device: MTLDevice) throws -> MTLBuffer {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw err("cannot open \(url.lastPathComponent)") }
        defer { close(fd) }
        let size = Int(lseek(fd, 0, SEEK_END))
        _ = lseek(fd, 0, SEEK_SET)
        guard size > 0, let buf = device.makeBuffer(length: size, options: .storageModeShared)
        else { throw err("makeBuffer failed for \(url.lastPathComponent) (\(size) bytes)") }
        var done = 0
        while done < size {
            let n = read(fd, buf.contents() + done, min(1 << 27, size - done))
            guard n > 0 else { throw err("read failed for \(url.lastPathComponent)") }
            done += n
        }
        return buf
    }

    /// "MEM <tag> footprint=X.XX GB headroom=Y.YY GB" — jetsam diagnostics
    /// (gemma tbl binds ~2.35 GB of owned tables; generation peaks ~4.4 GB
    /// against the entitled ~6.4 GB limit on a 12 GB iPhone).
    static func memLine(_ tag: String) -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        let footprint = kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : -1
        let headroom = Double(os_proc_available_memory()) / 1e9
        return String(format: "MEM %@ footprint=%.2f GB headroom=%.2f GB", tag, footprint, headroom)
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "PipelinedBackend", code: 1,
                userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
