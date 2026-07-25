// Gemma-4-E2B raw-Metal backend — the zoo's hand-written Metal decode loop
// (no Core AI engine, no .aimodel): mmap'd mixed-bit QAT pack (int2/int4/int8 +
// PLE tables, 2.0 GB) + 5 hand-tuned kernels compiled at load, greedy decode
// with on-GPU argmax. iPhone 17 Pro fresh decode ~55 tok/s = LiteRT-LM parity
// (same-afternoon interleaved A/B, 2026-07-15), lossless vs the fp16 oracle
// (S1 token gate).
//
// Like RWKV7Backend / SpecDecodeBackend this backend owns its whole loop:
// tokenizer (the bundled gemma tokenizer + applyChatTemplate), stop ids
// (<eos>=1, <turn|>=106), and streaming. The heavy lifting lives in
// Gemma4MetalEngine.swift (dispatch sequence copied verbatim from the gated
// bench runner — see the header there before touching anything).
//
// On-device layout (under Documents/models/gemma4_e2b_raw_metal/):
//   gemma4_pack.bin    — 2.18 GB mixed-bit weight blob (mmap'd, stays on disk)
//   gemma4_pack.json   — tensor manifest + model meta
// Kernels + oracle refs ride the app bundle (Resources/g4msl) — they must
// version with the host dispatch code, not with the weights.
import Foundation
import Tokenizers

@MainActor
final class Gemma4MetalBackend {
    nonisolated static let modelDir = "gemma4_e2b_raw_metal"
    nonisolated static let label = "Gemma 4 E2B ⚡raw-Metal"

    private var engine: Gemma4MetalEngine?
    private var tokenizer: Tokenizer?

    var loaded: Bool { engine != nil }
    var ctx: Int { engine?.maxContext ?? 0 }

    struct GenStats {
        var prefillTok: Int; var prefillSec: Double
        var decodeTok: Int; var decodeSec: Double
        var summary: String {
            String(format: "%@ · prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s",
                   Gemma4MetalBackend.label,
                   prefillTok, Double(prefillTok) / max(prefillSec, 1e-6),
                   decodeTok, Double(decodeTok) / max(decodeSec, 1e-6))
        }
    }

    // MARK: - Load

    func load() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let packDir = docs.appendingPathComponent("models").appendingPathComponent(Self.modelDir)
        guard let mslDir = Bundle.main.url(forResource: "g4msl", withExtension: nil) else {
            throw Self.err("g4msl kernel resources missing from the app bundle")
        }
        // Same tokenizer discipline as the app's other gemma modes: bundled tokenizer,
        // overridable by a sideloaded Documents/tokenizer.
        let sideloaded = docs.appendingPathComponent("tokenizer")
        let tokFolder = FileManager.default.fileExists(atPath: sideloaded.path)
            ? sideloaded
            : (Bundle.main.url(forResource: "tokenizer", withExtension: nil) ?? sideloaded)

        print("[g4metal] loading pack + compiling kernels …")
        let t0 = Date()
        let tok = try await AutoTokenizer.from(modelFolder: tokFolder, strict: false)
        // Off-main: pack mmap + 3x makeLibrary(source:) take a few seconds.
        let eng = try await Task.detached(priority: .userInitiated) {
            try Gemma4MetalEngine(packDir: packDir, mslDir: mslDir)
        }.value
        engine = eng; tokenizer = tok
        print(String(format: "[g4metal] ready in %.1fs · ctx %d", -t0.timeIntervalSinceNow, eng.maxContext))

        // Template selftest: the exact ids the losslessness proof used. A mismatch means
        // the tokenizer/scaffold drifted — generation would be off-oracle. Warn loudly.
        // (The bundled tokenizer's jinja template inserts an extra "\n\n" after <bos>,
        // so the backend hand-renders the turn scaffold instead — same as the kit's
        // GemmaPromptRenderer.)
        let want = [2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107]
        let got = Self.renderIds(prompt: "Why is the sky blue?", tokenizer: tok)
        if got != want {
            print("[g4metal] ⚠️ TEMPLATE SELFTEST MISMATCH got \(got) want \(want)")
        }

        // Headless S1 token gate (G4CHAT_GATE=1): greedy ids vs the fp16/bf16 oracle.
        if ProcessInfo.processInfo.environment["G4CHAT_GATE"] == "1" {
            let refs = mslDir.appendingPathComponent("oracle_refs.json")
            let r = try await Task.detached { try eng.tokenGate(refsURL: refs) }.value
            for line in r.detail { print("[g4metal] gate \(line)") }
            print("[g4metal] STATS S1_GATE \(r.pass ? "PASS" : "FAIL")")
        }
    }

    func unload() { engine = nil; tokenizer = nil }

    /// New conversation — drop the incremental KV state.
    func reset() { engine?.reset() }

    // MARK: - Generate

    /// Greedy chat generation, streaming decoded text via `onText` (full text so far).
    /// Single-turn like the app's other backends; the engine itself reuses the KV
    /// prefix across calls, so a repeated prompt prefix costs no re-prefill.
    /// The Gemma-4 turn scaffold, hand-rendered (matches the oracle prompt ids exactly:
    /// <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n; gemma's tokenizer adds no bos).
    nonisolated private static func renderIds(prompt: String, tokenizer: Tokenizer) -> [Int] {
        [2] + tokenizer.encode(text: "<|turn>user\n\(prompt)<turn|>\n<|turn>model\n")
    }

    @discardableResult
    func generate(_ prompt: String, maxNew: Int = 512,
                  onText: @escaping (String) -> Void) async throws -> GenStats {
        guard let engine, let tokenizer else { throw Self.err("not loaded") }
        let ids = Self.renderIds(prompt: prompt, tokenizer: tokenizer)
        guard ids.count + 16 < engine.maxContext else {
            throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(engine.maxContext)")
        }
        let stream = AsyncStream<[Int]> { cont in
            Task.detached(priority: .userInitiated) {
                engine.generate(promptIds: ids, maxNew: maxNew,
                                onTokens: { cont.yield($0) })
                cont.finish()
            }
        }
        var gen: [Int] = []
        for await batch in stream {
            gen.append(contentsOf: batch)
            onText(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
        onText(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        let s = engine.lastStats
        return GenStats(prefillTok: s.prefillTokens, prefillSec: s.prefillSeconds,
                        decodeTok: s.decodeTokens, decodeSec: s.decodeSeconds)
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "Gemma4MetalBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
