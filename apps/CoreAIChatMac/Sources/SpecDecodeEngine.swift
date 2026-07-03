// SpecDecodeEngine — two-model speculative decoding for the Qwen3.6-27B GDN hybrid.
//
// LOSSLESS greedy spec-decode: a 0.8B int4 draft proposes K tokens, the 27B verify
// graph checks them all in ONE static S-window forward. Every committed token equals
// the 27B's own greedy argmax, so ⚡Spec ON and OFF produce byte-identical text — the
// toggle only changes speed (Mac-measured: code ~2.1×, RAG ~1.7×, free-form ~1.9×
// projected; see knowledge/spec-decode-hybrid-verify-design.md).
//
// Unlike the dense iOS SpecDecodeBackend (2 KV states, rollback = don't advance
// `processed`), the 27B is a GDN HYBRID: 48/64 layers carry cumulative conv/rec
// states that cannot be rolled back. This engine ports the python-validated
// WindowedModel discipline (_spec_mac_two_model.py, 21 runs LOSSLESS 48/48):
//   * device conv/rec state only ever covers a fully-committed prefix C[:m];
//   * committed tokens past m ride as the TAIL of every feed;
//   * every feed is EXACTLY S tokens at positions m..m+S-1 (verify graph is a
//     static [1,S] query) — one runtime shape per anchor, not per token;
//   * a feed's state advance is kept only when the whole window was committed
//     tokens (all-accept rounds commit free); otherwise conv/rec restore from a
//     host snapshot and a later exact-S re-anchor forward moves m;
//   * KV rows for positions ≥ m are rewritten by every feed before they are
//     attended, so only conv/rec need the snapshot.
//
// Both bundles load with expectFrequentReshapes = true — without it the runtime
// tries a fixed-shape ANE compile per position length and dies on OS27
// (ANECompile → MTL4CommandQueueError; the python harness needed reload-every-3
// because the python API lacks this flag).
import CoreAI
import CoreAIShared
import Foundation
import Metal
import Tokenizers

@MainActor
final class SpecDecodeEngine {
    static let maxDraftK = 6      // frozen by the Mac K sweep (K>8 never pays; 6 ≈ 8 at S=9)
    private let maxNewTokens = 256

    private let target: WindowedModel
    private let draft: WindowedModel
    private let tokenizer: any Tokenizer
    private let stopIds: Set<Int32>

    // A spec bundle is a verify export (static S-window graph): metadata carries
    // "verify_query_len". Regular decode bundles / LanguageBundles don't.
    static func isSpecVerify(bundleURL: URL) -> Bool {
        (Self.metadata(bundleURL)?["verify_query_len"] as? Int) != nil
    }

    // The paired draft lives next to the target bundle: metadata "spec_draft" names
    // the sibling directory; otherwise take the smallest OTHER verify bundle in the
    // same folder (the draft is always the small one).
    static func findDraft(for targetURL: URL) -> URL? {
        let parent = targetURL.deletingLastPathComponent()
        if let name = Self.metadata(targetURL)?["spec_draft"] as? String {
            let url = parent.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil)) ?? []
        return entries
            .filter { $0.lastPathComponent != targetURL.lastPathComponent && isSpecVerify(bundleURL: $0) }
            .min { bundleSize($0) < bundleSize($1) }
    }

    private static func metadata(_ bundleURL: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: bundleURL.appendingPathComponent("metadata.json"))
        else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func bundleSize(_ url: URL) -> Int64 {
        let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var total: Int64 = 0
        while let file = enumerator?.nextObject() as? URL {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    init(targetURL: URL, draftURL: URL) async throws {
        target = try await WindowedModel(bundleURL: targetURL)
        draft = try await WindowedModel(bundleURL: draftURL)
        guard target.vocab == draft.vocab else {
            throw Self.err("draft vocab \(draft.vocab) ≠ target vocab \(target.vocab) — "
                           + "spec-decode needs a same-tokenizer draft")
        }
        tokenizer = try await AutoTokenizer.from(
            modelFolder: targetURL.appendingPathComponent("tokenizer"), strict: false)
        var stops: Set<Int32> = []
        for token in ["<|im_end|>", "<|endoftext|>"] {
            if let id = tokenizer.convertTokenToId(token) { stops.insert(Int32(id)) }
        }
        stopIds = stops
    }

    // MARK: - Chat prompt

    // Hand-built ChatML with an EMPTY pre-seeded <think> block (same as the shipped iOS
    // fast mode): the model answers directly instead of emitting a reasoning trace —
    // spec-decode is a speed feature, and draft acceptance is higher on direct answers.
    private func promptIds(history: [[String: any Sendable]]) -> [Int32] {
        var text = ""
        for message in history {
            let role = (message["role"] as? String) ?? "user"
            let content = (message["content"] as? String) ?? ""
            text += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tokenizer.encode(text: text).map(Int32.init)
    }

    // MARK: - Generation

    struct Stats {
        var promptTokens = 0
        var generated = 0
        var rounds = 0
        var acceptedDrafts = 0
        var targetForwards = 0
        var draftForwards = 0
        var note: String {
            guard rounds > 0 else { return "" }
            let alpha = Double(acceptedDrafts) / Double(rounds)
            let perForward = Double(generated) / Double(max(targetForwards, 1))
            return String(format: "α %.2f/round · %.2f tok/fwd", alpha, perForward)
        }
    }

    /// Greedy generation through the verify graph. `specOn=false` runs the exact same
    /// window discipline with no drafts (the lossless A/B baseline) — outputs are
    /// byte-identical either way, only the forward count changes.
    func generate(history: [[String: any Sendable]], specOn: Bool,
                  onUpdate: @MainActor (String, Stats) -> Void) async throws -> (text: String, stats: Stats) {
        // Fit prompt + generation inside the KV capacity; drop the oldest turns first.
        var msgs = history
        var ids = promptIds(history: msgs)
        let budgetCap = target.kvCapacity - target.s * 2 - maxNewTokens
        while ids.count > budgetCap && msgs.count > 1 {
            msgs.removeFirst()
            ids = promptIds(history: msgs)
        }
        guard ids.count > 1, ids.count <= budgetCap else {
            throw Self.err("prompt (\(ids.count) tok) does not fit the \(target.kvCapacity) KV window")
        }

        var stats = Stats(promptTokens: ids.count)
        target.resetStream()
        draft.resetStream()

        // Prefill both streams, holding the LAST prompt token back from the committed
        // stream: the anchor bootstrap peeks it, so the anchor is the prediction after
        // the true last token even when the tail is empty (a prompt length that is an
        // exact multiple of S would otherwise anchor off a pad row).
        let held = ids.removeLast()
        try await target.append(ids)
        let (bootFlat, bootBase) = try await target.peek([held])
        var pending = argmaxRow(bootFlat, row: bootBase, vocab: target.vocab)
        try await target.append([held])
        try await draft.append(ids + [held])

        var gen: [Int32] = []
        var lastEmit = SuspendingClock.now
        var stopped = false

        while gen.count < maxNewTokens && !stopped {
            if stopIds.contains(pending) { break }
            let k = specOn ? min(Self.maxDraftK, target.room()) : 0
            let drafts = k > 0 ? try await propose(k: k, anchor: pending) : []

            let (flat, base) = try await target.verifyRound(anchor: pending, drafts: drafts)
            var accepted = 0
            while accepted < drafts.count,
                  argmaxRow(flat, row: base + accepted, vocab: target.vocab) == drafts[accepted] {
                accepted += 1
            }
            let committed = [pending] + drafts.prefix(accepted)
            pending = argmaxRow(flat, row: base + accepted, vocab: target.vocab)
            try await target.commitOrRestore(accepted: committed, allDrafts: drafts.count)
            try await draft.append(committed)   // drafter absorbs the committed delta

            stats.rounds += 1
            stats.acceptedDrafts += accepted
            for token in committed {
                if stopIds.contains(token) { stopped = true; break }
                gen.append(token)
                if gen.count >= maxNewTokens { break }
            }
            stats.generated = gen.count
            stats.targetForwards = target.forwards
            stats.draftForwards = draft.forwards

            if SuspendingClock.now - lastEmit >= .milliseconds(40) {
                onUpdate(decode(gen), stats)
                lastEmit = SuspendingClock.now
                await Task.yield()
            }
        }

        stats.generated = gen.count
        stats.targetForwards = target.forwards
        stats.draftForwards = draft.forwards
        let text = decode(gen)
        onUpdate(text, stats)
        return (text, stats)
    }

    // Draft proposes greedily through its own S-window, seeded past the anchor so
    // proposals continue the stream the target actually verifies (C + [a0] + drafts).
    private func propose(k: Int, anchor: Int32) async throws -> [Int32] {
        let cap = min(k, draft.room())
        guard cap > 0 else { return [] }
        var drafts: [Int32] = []
        while drafts.count < cap {
            let (flat, base) = try await draft.peek([anchor] + drafts)
            drafts.append(argmaxRow(flat, row: base + drafts.count, vocab: draft.vocab))
        }
        return drafts
    }

    private func decode(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: true)
    }

    private func argmaxRow(_ flat: [Float], row: Int, vocab: Int) -> Int32 {
        let base = row * vocab
        var bestIndex = 0
        var bestValue = -Float.infinity
        for j in 0..<vocab where flat[base + j] > bestValue {
            bestValue = flat[base + j]
            bestIndex = j
        }
        return Int32(bestIndex)
    }

    fileprivate static func err(_ message: String) -> NSError {
        NSError(domain: "SpecDecodeEngine", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
    }
}

// MARK: - WindowedModel

// One verify bundle (static [1,S] query) + the committed-prefix state discipline.
// Mirrors the python WindowedModel that passed 21/21 LOSSLESS runs on this exact
// graph pair; see the file header for the invariants.
@MainActor
final class WindowedModel {
    let s: Int                    // verify window (query length), from bundle metadata
    let vocab: Int
    let kvCapacity: Int
    private(set) var forwards = 0

    private let fn: InferenceFunction
    private let desc: InferenceFunctionDescriptor
    private let inName: String
    private let posName: String
    private let logitsName: String
    private let padToken: Int32 = 0

    private var model: AIModel?                             // keeps the mapped weights alive
    private var states: [String: NDArray] = [:]
    private var snapshotStates: [String: [Float16]] = [:]   // conv/rec only — see header
    private(set) var stream: [Int32] = []                   // committed tokens C
    private var anchor = 0                                  // m: device state covers C[:m]
    private var roundDrafts = 0
    private var roundPad = 0

    init(bundleURL: URL) async throws {
        let metaData = try Data(contentsOf: bundleURL.appendingPathComponent("metadata.json"))
        let meta = (try? JSONSerialization.jsonObject(with: metaData)) as? [String: Any] ?? [:]
        guard let windowLen = meta["verify_query_len"] as? Int else {
            throw SpecDecodeEngine.err("\(bundleURL.lastPathComponent) is not a verify bundle")
        }
        s = windowLen
        // KV states are allocated at the export trace capacity (TRACE_KV_CACHE_SEQ_LEN),
        // not the tokenizer's max_context_length.
        kvCapacity = 2048

        // Prefer an AOT slice for THIS GPU (Metal "applegpu_g16s" → "<name>.h16s.aimodelc")
        // when one is present AND loadable — it ships every shape precompiled. Fall back to
        // the JIT .aimodel otherwise: anchors only ever advance in steps of S, so the shape
        // set is CLOSED (position lengths = multiples of S) and the runtime's on-disk
        // specialization cache makes repeat runs fast after a one-time warm-up. (On this
        // OS build the beta coreai-build's .aimodelc is rejected with invalidCompiledModel
        // — toolchain/runtime format skew — hence the fallback rather than a hard error.)
        let jitName = ((meta["assets"] as? [String: Any])?["main"] as? String)
            ?? "\(bundleURL.lastPathComponent).aimodel"
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true   // multi-position-length runtime — see header

        var loaded: AIModel?
        if let gpuArch = MTLCreateSystemDefaultDevice()?.architecture.name,
           let code = gpuArch.split(separator: "_").last.map(String.init), code.hasPrefix("g") {
            let aotName = jitName.replacingOccurrences(
                of: ".aimodel", with: ".h\(code.dropFirst()).aimodelc")
            let aotURL = bundleURL.appendingPathComponent(aotName)
            if FileManager.default.fileExists(atPath: aotURL.path) {
                loaded = try? await AIModel(contentsOf: aotURL, options: options)
            }
        }
        let model: AIModel
        if let loaded {
            model = loaded
        } else {
            model = try await AIModel(contentsOf: bundleURL.appendingPathComponent(jitName),
                                      options: options)
        }
        guard let loaded = try model.loadFunction(named: "main") else {
            throw SpecDecodeEngine.err("\(bundleURL.lastPathComponent): no 'main' function")
        }
        fn = loaded
        desc = loaded.descriptor
        guard desc.inputNames.count >= 2 else {
            throw SpecDecodeEngine.err("expected input_ids + position_ids inputs")
        }
        inName = desc.inputNames[0]
        posName = desc.inputNames[1]
        logitsName = desc.outputNames[0]
        guard case .ndArray(let logDesc)? = desc.outputDescriptor(of: logitsName),
              let v = logDesc.shape.last, v > 0 else {
            throw SpecDecodeEngine.err("no logits output")
        }
        vocab = v
        self.model = model
        try allocStates()
        snapshot()
    }

    // MARK: state plumbing

    private func allocStates() throws {
        states = [:]
        for name in desc.stateNames {
            guard case .ndArray(let sd)? = desc.stateDescriptor(of: name) else {
                throw SpecDecodeEngine.err("state descriptor \(name)")
            }
            let shape = sd.shape.map { $0 < 0 ? kvCapacity : $0 }
            var array = NDArray(descriptor: sd.resolvingDynamicDimensions(shape))
            fillNDArray(&array, as: Float16.self,
                        with: [Float16](repeating: 0, count: shape.reduce(1, *)))
            states[name] = array
        }
    }

    /// Fresh stream (per chat turn): zeroed states, empty committed stream.
    func resetStream() {
        try? allocStates()
        stream = []
        anchor = 0
        snapshot()
    }

    // Only the cumulative GDN states need snapshotting; KV rows at positions ≥ m are
    // rewritten by every feed before they can be attended.
    private var cumulativeStateNames: [String] {
        desc.stateNames.filter { $0.localizedCaseInsensitiveContains("conv")
            || $0.localizedCaseInsensitiveContains("rec") }
    }

    private func snapshot() {
        for name in cumulativeStateNames {
            guard let array = states[name] else { continue }
            let count = array.shape.reduce(1, *)
            snapshotStates[name] = readNDArray(array, as: Float16.self, count: count)
        }
    }

    private func restore() {
        for name in cumulativeStateNames {
            guard var array = states[name], let saved = snapshotStates[name] else { continue }
            fillNDArray(&array, as: Float16.self, with: saved)
            states[name] = array
        }
    }

    // MARK: forward

    private func forward(_ tokens: [Int32]) async throws -> [Float] {
        precondition(tokens.count == s, "window feed must be exactly S tokens")
        guard case .ndArray(let inDesc)? = desc.inputDescriptor(of: inName),
              case .ndArray(let posDesc)? = desc.inputDescriptor(of: posName),
              case .ndArray(let logDesc)? = desc.outputDescriptor(of: logitsName) else {
            throw SpecDecodeEngine.err("descriptor")
        }
        let positions = anchor + s
        var idArray = NDArray(descriptor: inDesc.resolvingDynamicDimensions([1, s]))
        fillNDArray(&idArray, as: Int32.self, with: tokens)
        var posArray = NDArray(descriptor: posDesc.resolvingDynamicDimensions([1, positions]))
        fillNDArray(&posArray, as: Int32.self, with: (0..<positions).map(Int32.init))
        var logits = NDArray(descriptor: logDesc.resolvingDynamicDimensions([1, s, vocab]))

        // States threaded like the iOS backend: distinct locals in, run, write back.
        // MutableViews is ~Escapable — its lifetime depends on each inserted borrow, and
        // an array-element borrow (&arr[i]) ends at the statement, so only plain local
        // vars satisfy the checker. The qwen3_5 hybrid family always has exactly 4 states
        // (keyCache, valueCache, convState, recState).
        let names = desc.stateNames
        guard names.count == 4,
              var s0 = states[names[0]], var s1 = states[names[1]],
              var s2 = states[names[2]], var s3 = states[names[3]] else {
            throw SpecDecodeEngine.err("expected 4 hybrid states, got \(desc.stateNames.count)")
        }
        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&s0, for: names[0])
        stateViews.insert(&s1, for: names[1])
        stateViews.insert(&s2, for: names[2])
        stateViews.insert(&s3, for: names[3])
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logits, for: logitsName)
        _ = try await fn.run(inputs: [inName: idArray, posName: posArray],
                             states: consume stateViews, outputViews: consume outputViews)
        states[names[0]] = s0
        states[names[1]] = s1
        states[names[2]] = s2
        states[names[3]] = s3
        forwards += 1
        return flattenAsFloat(logits)
    }

    // MARK: committed-stream ops (python WindowedModel one-to-one)

    private var tail: [Int32] { Array(stream[anchor...]) }

    func room() -> Int { s - (stream.count - anchor) - 1 }

    private func reanchorIfFull() async throws {
        while stream.count - anchor >= s {
            _ = try await forward(Array(stream[anchor..<(anchor + s)]))   // all-committed feed
            anchor += s
            snapshot()
        }
    }

    func append(_ tokens: [Int32]) async throws {
        stream.append(contentsOf: tokens)
        try await reanchorIfFull()
    }

    /// Logits for feed = tail+extra+pad, discarding the state advance. Row `base-1+i`
    /// is the prediction after extra[i-1].
    func peek(_ extra: [Int32]) async throws -> (flat: [Float], base: Int) {
        let t = tail
        precondition(t.count + extra.count <= s, "peek overflows the window")
        let feed = t + extra + [Int32](repeating: padToken, count: s - t.count - extra.count)
        let flat = try await forward(feed)
        restore()
        return (flat, t.count)
    }

    /// One verify forward: feed = tail+[a0]+drafts+pad. Caller decides acceptance and
    /// then calls commitOrRestore.
    func verifyRound(anchor a0: Int32, drafts: [Int32]) async throws -> (flat: [Float], base: Int) {
        let t = tail
        precondition(t.count + 1 + drafts.count <= s, "verify round overflows the window")
        roundDrafts = drafts.count
        roundPad = s - t.count - 1 - drafts.count
        let feed = t + [a0] + drafts + [Int32](repeating: padToken, count: roundPad)
        let flat = try await forward(feed)
        return (flat, t.count)
    }

    /// Keep the round's state advance only when the whole window was committed tokens
    /// (every draft accepted, no pad); otherwise restore and let re-anchor catch up.
    func commitOrRestore(accepted: [Int32], allDrafts: Int) async throws {
        stream.append(contentsOf: accepted)
        if accepted.count == 1 + roundDrafts && roundDrafts == allDrafts && roundPad == 0 {
            anchor += s
            snapshot()
        } else {
            restore()
        }
        try await reanchorIfFull()
    }
}
