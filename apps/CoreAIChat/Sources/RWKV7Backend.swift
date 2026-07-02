// RWKV-7 "Goose" backend — zoo's first pure-recurrent / linear-attention LLM on device.
// Community port — NOT an Apple model.
//
// RWKV-7 has NO attention and NO KV cache: every layer is a WKV7 delta-rule matrix-state
// time-mix + sqrelu channel-mix. The decode .aimodelc is a fixed-shape S=1 step carrying only
// TWO states — recState[24,1,32,64,64] (the WKV7 matrix S) and shiftState[24,1,2,2048]
// (token-shift prev hidden). That is the O(1)-per-token win: constant memory, unbounded
// context, no growing KV.
//
// The shipped CoreAIPipelinedEngine can't drive this (it positionally assumes states[0]=keyCache,
// states[1]=valueCache and always grows a KV cache). So — exactly like BitVLABackend — we drive
// the low-level InferenceFunction directly, binding our two states BY NAME via MutableViews and
// looping S=1. Validated byte-for-byte by conversion/rwkv7/gate.py (teacher-forced top-1 127/127).
//
// On-device layout (under Documents/models/<modelDir>/):
//   <modelDir>.h18p.aimodelc   — AOT decode graph (compile: coreai-build compile … --platform iOS
//                                --architecture h18p --preferred-compute gpu). int8keepproj ≈ 2 GB.
//   rwkv_vocab.tsv             — base64 World vocab (conversion/rwkv7/prep_vocab.py).
//
// Register in Gemma4ChatEngine like BitVLA: add a GemmaMode.rwkv7 case and, in load(), branch to
// `let b = RWKV7Backend(); try await b.load(); rwkv7 = b`. Stream via generate(prompt:onToken:).
import CoreAI
import CoreAIShared
import Foundation

@MainActor
final class RWKV7Backend {
    static let modelDir = "rwkv7_goose_1_5b"
    static let vocabName = "rwkv_vocab.tsv"
    static let label = "RWKV-7 ⚡recurrent O(1)"

    private let NL = 24, NH = 32, HD = 64, H = 2048, VOCAB = 65536

    private var model: AIModel?
    private var fn: InferenceFunction?
    private var d: InferenceFunctionDescriptor?
    private var tokenizer: RWKVWorldTokenizer?
    // The two recurrent states persist across S=1 steps; zeroed per conversation turn.
    private var rec: NDArray?
    private var shift: NDArray?

    var loaded: Bool { fn != nil }

    struct GenStats { var prefillSec: Double; var decodeTok: Int; var decodeSec: Double }

    // MARK: - Load

    func load() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("models").appendingPathComponent(Self.modelDir)
        let url = Self.modelURL(dir, Self.modelDir)
        var opt = SpecializationOptions(preferredComputeUnitKind: .gpu)
        opt.expectFrequentReshapes = false   // fixed-shape S=1 graph

        print("[rwkv7] loading \(url.lastPathComponent) ...")
        let m = try await AIModel(contentsOf: url, options: opt)
        guard let f = try m.loadFunction(named: "main") else { throw Self.err("main missing") }
        model = m; fn = f; d = f.descriptor
        tokenizer = try RWKVWorldTokenizer(vocabURL: dir.appendingPathComponent(Self.vocabName))
        try allocStates()
        print(PipelinedBackend.memLine("\(Self.label) loaded"))
    }

    func unload() { fn = nil; model = nil; tokenizer = nil; rec = nil; shift = nil }

    private static func modelURL(_ dir: URL, _ name: String) -> URL {
        // The decode graph is best AOT-precompiled for the device GPU (h18p); fall back to the
        // plain .aimodel if no AOT package is present.
        let aot = dir.appendingPathComponent("\(name).h18p.aimodelc")
        if FileManager.default.fileExists(atPath: aot.path) { return aot }
        return dir.appendingPathComponent("\(name).aimodel")
    }

    /// (Re)allocate zeroed recurrent states — call at the start of each conversation turn.
    func resetState() throws { try allocStates() }

    private func allocStates() throws {
        guard let d else { throw Self.err("not loaded") }
        guard case .ndArray(let rd)? = d.stateDescriptor(of: "recState"),
              case .ndArray(let sd)? = d.stateDescriptor(of: "shiftState") else {
            throw Self.err("rec/shift state descriptors missing")
        }
        let rShape = [NL, 1, NH, HD, HD]
        let sShape = [NL, 1, 2, H]
        var r = NDArray(descriptor: rd.resolvingDynamicDimensions(rShape))
        var s = NDArray(descriptor: sd.resolvingDynamicDimensions(sShape))
        fillNDArray(&r, as: Float16.self, with: [Float16](repeating: 0, count: rShape.reduce(1, *)))
        fillNDArray(&s, as: Float16.self, with: [Float16](repeating: 0, count: sShape.reduce(1, *)))
        rec = r; shift = s
    }

    // MARK: - Decode

    /// One S=1 step: token id -> logits[VOCAB]. rec/shift mutate in place.
    private func step(_ token: Int) async throws -> [Float] {
        guard let fn, let d, var r = rec, var s = shift else { throw Self.err("not loaded") }
        guard case .ndArray(let iid)? = d.inputDescriptor(of: "input_ids"),
              case .ndArray(let pid)? = d.inputDescriptor(of: "position_ids"),
              case .ndArray(let od)? = d.outputDescriptor(of: "logits") else {
            throw Self.err("io descriptors missing")
        }
        var ii = NDArray(descriptor: iid.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&ii, as: Int32.self, with: [Int32(token)])
        var pp = NDArray(descriptor: pid.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&pp, as: Int32.self, with: [0])   // RWKV-7 is positionless; placeholder
        var out = NDArray(descriptor: od.resolvingDynamicDimensions([1, 1, VOCAB]))

        var states = InferenceFunction.MutableViews()
        states.insert(&r, for: "recState")
        states.insert(&s, for: "shiftState")
        var views = InferenceFunction.MutableViews()
        views.insert(&out, for: "logits")
        _ = try await fn.run(inputs: ["input_ids": ii, "position_ids": pp],
                             states: consume states, outputViews: consume views)
        rec = r; shift = s
        return flattenAsFloat(out)
    }

    /// Greedy chat generation. World chat format: <eot> + "User: {prompt}\n\nAssistant:".
    /// Streams decoded text deltas via `onToken` (decoding the full id list each step keeps
    /// multi-byte chars split across tokens intact). Returns prefill/decode timing.
    @discardableResult
    func generate(prompt: String, maxNewTokens: Int = 512,
                  onToken: @escaping (String) -> Void) async throws -> GenStats {
        guard let tokenizer else { throw Self.err("not loaded") }
        try resetState()

        var ids = [RWKVWorldTokenizer.eosToken]
        ids.append(contentsOf: tokenizer.encode("User: \(prompt)\n\nAssistant:"))

        let t0 = SuspendingClock.now
        var logits: [Float] = []
        for t in ids { logits = try await step(t) }
        let prefillSec = Self.seconds(since: t0)

        let tGen = SuspendingClock.now
        var genIds: [Int] = []
        var emitted = ""
        var decodeTok = 0
        for _ in 0..<maxNewTokens {
            let next = Self.argmax(logits)
            if next == RWKVWorldTokenizer.eosToken { break }
            genIds.append(next)
            decodeTok += 1
            let full = tokenizer.decode(genIds)
            if full.count > emitted.count {
                let delta = String(full[full.index(full.startIndex, offsetBy: emitted.count)...])
                onToken(delta)
                emitted = full
            }
            logits = try await step(next)
        }
        return GenStats(prefillSec: prefillSec, decodeTok: decodeTok,
                        decodeSec: Self.seconds(since: tGen))
    }

    // MARK: - Helpers

    private static func argmax(_ v: [Float]) -> Int {
        var bi = 0, bv = -Float.infinity
        for i in 0..<v.count where v[i] > bv { bv = v[i]; bi = i }
        return bi
    }

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let (secs, atto) = (SuspendingClock.now - start).components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "RWKV7Backend", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
