// SpecDecodeBackend — greedy n-gram (prompt-lookup) speculative decoding on the Core AI
// pipelined runtime. Stream C ship path. Community accel lever — NOT an Apple model.
//
// Loads a STATEFUL dense dynamic bundle (exports/qwen3_4b_gpu, AOT h18p: input_ids +
// position_ids, 2 KV states, logits [1, q, vocab] at ALL positions) directly via
// InferenceFunction.run — exactly like RWKV7Backend — and runs the device-validated FUSED
// greedy verify loop (ondevice/PipelinedBench SpeculativeDecoder.swift): each round forwards
// [a0, draft_0 … draft_{K-1}] where a0 = argmax(cachedLogits) is the guaranteed greedy next
// token; the one forward writes a0's KV, verifies the drafts (per-position argmax) AND yields
// the next round's cached logits. Accept the maximal matching draft prefix; rollback is free
// (advance `processed` past accepted tokens — rejected draft KV rows are overwritten by the next
// forward and never attended, since position_ids length bounds attention).
//
// Lossless (greedy): every committed token equals plain-greedy argmax, so Spec ON and Spec OFF
// produce byte-identical text — the toggle only changes speed. Device-measured real α (iPhone 17
// Pro, real BPE prompts): 4B code 2.35 accepted/round → 2.12×, RAG 1.40 → 2.08×. α is
// workload-dependent (prompt-lookup only wins when the output echoes input n-grams: code / RAG /
// structured), so a hit-rate gate disables the drafter after a run of misses — free-form chat is
// never slowed by wasted verifies, and at α≈0 the fused scheme still costs ~0.
//
// Two non-obvious device gotchas (both crash ANECompile → MTL4 on the OS27 beta if missed):
//   1. the verify graph MUST be AOT (.h18p.aimodelc) — a JIT dynamic-query graph dies.
//   2. load with SpecializationOptions.expectFrequentReshapes = true (+ .gpu) — the multi-shape
//      prefill/decode/verify runtime specialization otherwise tries a fixed-shape ANE compile.
//
// Register in Gemma4ChatEngine like RWKV7: a GemmaMode.qwen3spec case + a load() branch
// `let b = SpecDecodeBackend(); try await b.load(); specDecode = b`. Stream via
// generate(_:spec:maxNew:onUpdate:). Greedy-only (chat sampling would need the modified
// rejection-sampling verify). State handling mirrors RWKV7Backend (states held as instance
// properties, copied in/out per forward) so nothing is held `inout` across an await.
import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class SpecDecodeBackend {
    static let modelDir = "qwen3_4b_gpu"
    static let label = "Qwen3-4B ⚡Spec"

    // n-gram drafter knobs (device-tuned in the PipelinedBench probe).
    private let K = 8          // max draft length per round
    private let ngram = 3      // prompt-lookup n-gram order
    private let maxSeq = 2048  // KV capacity (bucket) — prompt + gen must fit
    private let gateAfter = 6  // disable drafting after this many consecutive zero-accept rounds

    private var model: AIModel?
    private var fn: InferenceFunction?
    private var d: InferenceFunctionDescriptor?
    private var tokenizer: Tokenizer?
    private var vocab = 151_936
    private var stateNames: [String] = []
    private var inName = "", posName = "", logitsName = ""
    private let stopTokens: Set<Int> = [151_643, 151_645]  // <|endoftext|>, <|im_end|>

    // The two KV states persist across forwards; re-zeroed per generation. Held as properties
    // (copied in/out inside forward) so no NDArray is held `inout` across an await — same shape as
    // RWKV7Backend's rec/shift.
    private var kv0: NDArray?
    private var kv1: NDArray?

    var loaded: Bool { fn != nil }
    private(set) var ctx = 2048

    struct GenStats {
        var spec = false
        var prefillTok = 0
        var prefillSec = 0.0
        var decodeTok = 0
        var decodeSec = 0.0
        var forwards = 0
        var acceptedDraft = 0
        var summary: String {
            let tps = Double(decodeTok) / max(decodeSec, 1e-6)
            let pre = String(format: "prefill %d tok %.1f tok/s", prefillTok,
                             Double(prefillTok) / max(prefillSec, 1e-6))
            if spec {
                let accPerRound = forwards > 0 ? Double(acceptedDraft) / Double(forwards) : 0
                return String(format: "Qwen3-4B ⚡Spec ON (lossless) · %@ | decode %d tok %.1f tok/s · accepted/round %.2f",
                              pre, decodeTok, tps, accPerRound)
            }
            return String(format: "Qwen3-4B greedy (Spec OFF) · %@ | decode %d tok %.1f tok/s",
                          pre, decodeTok, tps)
        }
    }

    // MARK: - Load

    func load() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("models").appendingPathComponent(Self.modelDir)
        // The verify bundle: an AOT *.aimodelc wins, else a JIT *.aimodel (same discovery as the
        // PB_SPEC probe — load the model directly, no LanguageBundle).
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?.sorted() ?? []
        guard let bundleName = entries.first(where: { $0.hasSuffix(".aimodelc") })
            ?? entries.first(where: { $0.hasSuffix(".aimodel") }) else {
            throw Self.err("no .aimodelc/.aimodel in \(Self.modelDir) (sideload the dense AOT bundle)")
        }
        var opt = SpecializationOptions(preferredComputeUnitKind: .gpu)
        opt.expectFrequentReshapes = true   // multi-shape prefill/decode/verify — see header gotcha 2

        print("[spec] loading \(bundleName) …")
        let m = try await AIModel(contentsOf: dir.appendingPathComponent(bundleName), options: opt)
        guard let f = try m.loadFunction(named: "main") else { throw Self.err("main function missing") }
        let desc = f.descriptor
        guard desc.inputNames.count >= 2 else { throw Self.err("expected input_ids + position_ids") }
        guard desc.stateNames.count == 2 else {
            throw Self.err("expected 2 KV states (dense), got \(desc.stateNames.count)")
        }
        model = m; fn = f; d = desc
        inName = desc.inputNames[0]
        posName = desc.inputNames[1]
        logitsName = desc.outputNames[0]
        stateNames = desc.stateNames
        if case .ndArray(let logD)? = desc.outputDescriptor(of: logitsName) {
            vocab = logD.shape.last ?? vocab
        }
        // Tokenizer from the bundle's own HF tokenizer/ dir (tokenizer.json + config).
        tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"),
                                                 strict: false)
        ctx = maxSeq
        print(PipelinedBackend.memLine("\(Self.label) loaded"))
    }

    func unload() { fn = nil; model = nil; d = nil; tokenizer = nil; kv0 = nil; kv1 = nil }

    // MARK: - Model forward (stateful, 2 KV states)

    /// (Re)allocate zeroed KV states — call at the start of each generation.
    private func resetStates() throws {
        guard let d else { throw Self.err("not loaded") }
        func mk(_ name: String) throws -> NDArray {
            guard case .ndArray(let sd)? = d.stateDescriptor(of: name) else {
                throw Self.err("state descriptor \(name)")
            }
            let resolved = sd.shape.map { $0 < 0 ? maxSeq : $0 }
            let cnt = resolved.reduce(1, *)
            var arr = NDArray(descriptor: sd.resolvingDynamicDimensions(resolved))
            fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0, count: cnt))
            return arr
        }
        kv0 = try mk(stateNames[0])
        kv1 = try mk(stateNames[1])
    }

    // One forward: input_ids = [1, q], position_ids = [0 … processed+q-1], logits = [1, q, vocab]
    // (ALL positions). The 2 KV states are threaded in place (grown to processed+q by the graph).
    private func forward(_ tokens: [Int32], processed: Int) async throws -> [Float] {
        guard let fn, let d, var s0 = kv0, var s1 = kv1 else { throw Self.err("not loaded") }
        let q = tokens.count
        guard case .ndArray(let inD)? = d.inputDescriptor(of: inName),
              case .ndArray(let posD)? = d.inputDescriptor(of: posName),
              case .ndArray(let logD)? = d.outputDescriptor(of: logitsName) else { throw Self.err("desc") }
        var idArr = NDArray(descriptor: inD.resolvingDynamicDimensions([1, q]))
        fillNDArray(&idArr, as: Int32.self, with: tokens)
        var posArr = NDArray(descriptor: posD.resolvingDynamicDimensions([1, processed + q]))
        fillNDArray(&posArr, as: Int32.self, with: (0..<(processed + q)).map { Int32($0) })
        var outArr = NDArray(descriptor: logD.resolvingDynamicDimensions([1, q, vocab]))

        var sv = InferenceFunction.MutableViews()
        sv.insert(&s0, for: stateNames[0])
        sv.insert(&s1, for: stateNames[1])
        var ov = InferenceFunction.MutableViews()
        ov.insert(&outArr, for: logitsName)
        _ = try await fn.run(inputs: [inName: idArr, posName: posArr],
                             states: consume sv, outputViews: consume ov)
        kv0 = s0; kv1 = s1
        return flattenAsFloat(outArr)
    }

    private func argmaxRow(_ flat: [Float], row: Int) -> Int32 {
        let base = row * vocab
        var bi = 0
        var bv = -Float.infinity
        for j in 0..<vocab where flat[base + j] > bv { bv = flat[base + j]; bi = j }
        return Int32(bi)
    }

    // n-gram / prompt-lookup: most-recent earlier occurrence of the last `ng` tokens → its
    // continuation (backing off to shorter n-grams). Zero model cost.
    private func draft(_ seq: [Int32], _ cap: Int) -> [Int32] {
        let n = seq.count
        var ng = ngram
        while ng >= 1 {
            if n >= ng {
                let key = Array(seq[(n - ng)..<n])
                var i = n - ng - 1
                while i >= 0 {
                    if Array(seq[i..<(i + ng)]) == key {
                        let start = i + ng
                        let end = min(start + cap, n)
                        if end > start { return Array(seq[start..<end]) }
                        break
                    }
                    i -= 1
                }
            }
            ng -= 1
        }
        return []
    }

    // MARK: - Generate (spec ON = n-gram verify, OFF = plain greedy), streaming decoded text.

    @discardableResult
    func generate(
        _ prompt: String, spec: Bool, maxNew: Int, onUpdate: @escaping (String) -> Void
    ) async throws -> GenStats {
        guard let tokenizer else { throw Self.err("not loaded") }
        let ids = templatedIds(prompt, tokenizer: tokenizer)
        guard ids.count < maxSeq - K - 2 else {
            throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(maxSeq)")
        }
        let budget = min(maxNew, maxSeq - ids.count - K - 2)

        try resetStates()
        var stats = GenStats(spec: spec, prefillTok: ids.count)

        // Prefill (one forward over the whole prompt).
        let t0 = SuspendingClock.now
        let pre = try await forward(ids, processed: 0)
        var processed = ids.count
        stats.prefillSec = Self.seconds(since: t0)

        var seq = ids
        var gen: [Int] = []
        var cached = Array(pre[(pre.count - vocab)..<pre.count])   // logits at position `processed`
        var missStreak = 0

        let tGen = SuspendingClock.now
        var uiSec = 0.0
        var lastEmit = tGen
        func emit() {
            let u = SuspendingClock.now
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
            uiSec += Self.seconds(since: u)
            lastEmit = SuspendingClock.now
        }

        while gen.count < budget {
            let a0 = argmaxRow(cached, row: 0)                     // guaranteed greedy next token
            if stopTokens.contains(Int(a0)) { break }
            // Draft only when spec is on and the drafter hasn't gone cold (hit-rate gate).
            let drafts = (spec && missStreak < gateAfter) ? draft(seq + [a0], K) : []
            let verifyTokens = [a0] + drafts

            let flat = try await forward(verifyTokens, processed: processed)
            stats.forwards += 1
            var accepted = 0
            for i in 0..<drafts.count {
                if drafts[i] == argmaxRow(flat, row: i) { accepted += 1 } else { break }
            }
            stats.acceptedDraft += accepted
            missStreak = accepted == 0 ? missStreak + 1 : 0

            // Commit a0 + accepted drafts (stop at an EOS/stop token).
            var committed = [a0]
            committed.append(contentsOf: drafts[0..<accepted])
            var stopped = false
            for t in committed {
                if stopTokens.contains(Int(t)) { stopped = true; break }
                seq.append(t)
                gen.append(Int(t))
                if gen.count >= budget { break }
            }
            processed += committed.count
            // Next round's cached logits = row `accepted` (distribution at the new `processed`).
            cached = Array(flat[(accepted * vocab)..<((accepted + 1) * vocab)])

            if SuspendingClock.now - lastEmit >= .milliseconds(40) { emit() }
            if stopped { break }
        }
        let tFlush = SuspendingClock.now
        onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))   // final flush
        uiSec += Self.seconds(since: tFlush)
        stats.decodeTok = gen.count
        stats.decodeSec = Self.seconds(since: tGen) - uiSec
        print(PipelinedBackend.memLine("\(Self.label) gen spec=\(spec)"))
        return stats
    }

    // Qwen3 ChatML with an EMPTY pre-seeded <think> block, so the model skips the reasoning trace
    // and answers directly. This "fast mode" is deliberately no-think: reasoning traces are novel
    // text with ~zero n-gram echo (they drive prompt-lookup α to 0), and a fast lossless mode wants
    // concise answers anyway. Seeded at the token level because swift-transformers' template here
    // doesn't honor the Qwen3 /no_think soft switch. The win shows on input-grounded output
    // (extractive / code / structured), where the answer reuses the prompt's n-grams.
    private func templatedIds(_ prompt: String, tokenizer: Tokenizer) -> [Int32] {
        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tokenizer.encode(text: text).map { Int32($0) }
    }

    // MARK: - Helpers

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let (secs, atto) = (SuspendingClock.now - start).components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "SpecDecodeBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
