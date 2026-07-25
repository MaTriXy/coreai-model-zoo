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
        target = try await WindowedModel(bundleURL: targetURL, role: "target")
        draft = try await WindowedModel(bundleURL: draftURL, role: "draft")
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
    func generate(history: [[String: any Sendable]], specOn specOnRequested: Bool,
                  onUpdate: @MainActor (String, Stats) -> Void) async throws -> (text: String, stats: Stats) {
        // SPEC_OFF forces the no-draft baseline (pure greedy through the verify graph) so a
        // bench run can A/B the two paths for byte-identical output — bench only.
        let specOn = specOnRequested && ProcessInfo.processInfo.environment["SPEC_OFF"] == nil
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
        let bootBase = try await target.peek([held])
        var pending = target.argmax(row: bootBase)
        try await target.append([held])
        try await draft.append(ids + [held])

        var gen: [Int32] = []
        var lastEmit = SuspendingClock.now
        var stopped = false

        // Decode-only timer (excludes prefill/anchor bootstrap) for the SPEC_BENCH A/B.
        let decodeStart = SuspendingClock.now

        while gen.count < maxNewTokens && !stopped {
            if stopIds.contains(pending) { break }
            let k = specOn ? min(Self.maxDraftK, target.room()) : 0
            let drafts = k > 0 ? try await propose(k: k, anchor: pending) : []

            let base = try await target.verifyRound(anchor: pending, drafts: drafts)
            var accepted = 0
            while accepted < drafts.count,
                  target.argmax(row: base + accepted) == drafts[accepted] {
                accepted += 1
            }
            let committed = [pending] + drafts.prefix(accepted)
            pending = target.argmax(row: base + accepted)
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

        // SPEC_BENCH: emit a decode-only line (tok/s comparable to the shipped 15.9 decode).
        if ProcessInfo.processInfo.environment["SPEC_BENCH"] != nil {
            let dt = SuspendingClock.now - decodeStart
            let secs = Double(dt.components.seconds) + Double(dt.components.attoseconds) / 1e18
            let tps = secs > 0 ? Double(gen.count) / secs : 0
            let alpha = stats.rounds > 0 ? Double(stats.acceptedDrafts) / Double(stats.rounds) : 0
            let perFwd = Double(gen.count) / Double(max(stats.targetForwards, 1))
            FileHandle.standardError.write(Data(String(
                format: "SPEC STATS spec=%@ gen=%d decode_s=%.3f decode_tps=%.2f alpha=%.2f tok/fwd=%.2f tgt_fwd=%d draft_fwd=%d\n",
                specOn ? "on" : "off", gen.count, secs, tps, alpha, perFwd,
                stats.targetForwards, stats.draftForwards).utf8))
        }

        let text = decode(gen)
        if ProcessInfo.processInfo.environment["SPEC_BENCH"] != nil {
            FileHandle.standardError.write(Data("SPEC TEXT[\(specOn ? "on" : "off")]<<<\(text)>>>\n".utf8))
        }
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
            let base = try await draft.peek([anchor] + drafts)
            drafts.append(draft.argmax(row: base + drafts.count))
        }
        return drafts
    }

    private func decode(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: true)
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
    let role: String              // "target" / "draft" — SPEC_BENCH per-forward labelling
    private(set) var forwards = 0
    private let benchForwards = ProcessInfo.processInfo.environment["SPEC_BENCH"] != nil

    private let fn: InferenceFunction
    private let desc: InferenceFunctionDescriptor
    private let inName: String
    private let posName: String
    private let logitsName: String
    private let padToken: Int32 = 0

    // Owned-buffer forward path (mirrors CoreAIPipelinedEngine): every forward binds the
    // SAME persistent MTLBuffers via function.encode(...to: computeStream) so Core AI never
    // re-allocates per-call scratch — this is the 135ms(raw fn.run) → 63ms(engine kernels)
    // win that makes the loop beat the shipped 15.9 tok/s decode. The stream is drained
    // (currentWorkCompleted) after each encode because the spec loop is strictly sequential:
    // a round can't propose until it has read the previous round's logits.
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    // ComputeStream is a non-Sendable final class; currentWorkCompleted() suspends, which
    // would "send" it off @MainActor. Access is serialized (the spec loop awaits each forward
    // to completion before the next), so the stream is never touched concurrently.
    nonisolated(unsafe) private let computeStream: ComputeStream

    private let idBuffer: MTLBuffer          // [1, s]  Int32 — query tokens (fixed S)
    private let posBuffer: MTLBuffer         // [1, kvCapacity] Int32 — pre-filled 0..<kvCapacity
    private let logitsBuffer: MTLBuffer      // [1, s, vocab] Float16
    private let idScalar: NDArray.ScalarType
    private let posScalar: NDArray.ScalarType
    private let inDesc: NDArrayDescriptor
    private let posDesc: NDArrayDescriptor
    private let logitsDesc: NDArrayDescriptor

    // Model states as owned buffers (KV pair + GDN conv/rec). Distinct locals are rebuilt
    // per forward for the MutableViews lifetime rule; the buffers persist.
    private struct StateBinding {
        let name: String
        let buffer: MTLBuffer
        let scalarType: NDArray.ScalarType
        let shape: [Int]
        let strides: [Int]
        let byteCount: Int
    }
    private var stateBindings: [StateBinding] = []
    private var snapshotStates: [String: [UInt8]] = [:]     // conv/rec only — see header

    private var model: AIModel?                             // keeps the mapped weights alive
    private(set) var stream: [Int32] = []                   // committed tokens C
    private var anchor = 0                                  // m: device state covers C[:m]
    private var roundDrafts = 0
    private var roundPad = 0

    init(bundleURL: URL, role: String) async throws {
        self.role = role
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
        guard case .ndArray(let inputDesc)? = desc.inputDescriptor(of: inName),
              case .ndArray(let positionDesc)? = desc.inputDescriptor(of: posName),
              case .ndArray(let logDesc)? = desc.outputDescriptor(of: logitsName),
              let v = logDesc.shape.last, v > 0 else {
            throw SpecDecodeEngine.err("missing input_ids/position_ids/logits descriptors")
        }
        vocab = v
        self.model = model
        inDesc = inputDesc
        posDesc = positionDesc
        logitsDesc = logDesc
        idScalar = inputDesc.scalarType
        posScalar = positionDesc.scalarType

        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else {
            throw SpecDecodeEngine.err("no Metal device/command queue")
        }
        queue.label = "SpecDecode.\(bundleURL.lastPathComponent)"
        device = dev
        commandQueue = queue
        computeStream = ComputeStream(commandQueue: queue)

        // Persistent input/position/logits buffers, sized once: S is fixed and positions
        // never exceed kvCapacity, so the shape set is closed and Core AI reuses them.
        let idBytes = inputDesc.resolvingDynamicDimensions([1, s]).minimumByteCount
        let posBytes = positionDesc.resolvingDynamicDimensions([1, kvCapacity]).minimumByteCount
        let logitsBytes = logDesc.resolvingDynamicDimensions([1, s, v]).minimumByteCount
        guard let idBuf = dev.makeBuffer(length: idBytes, options: .storageModeShared),
              let posBuf = dev.makeBuffer(length: posBytes, options: .storageModeShared),
              let logitsBuf = dev.makeBuffer(length: logitsBytes, options: .storageModeShared) else {
            throw SpecDecodeEngine.err("input/logits buffer allocation failed")
        }
        idBuffer = idBuf
        logitsBuffer = logitsBuf
        // position_ids are the constant ramp 0,1,2,… — write once, bind a [1, positions] prefix.
        let posPtr = posBuf.contents().bindMemory(to: Int32.self, capacity: kvCapacity)
        for i in 0..<kvCapacity { posPtr[i] = Int32(i) }
        posBuffer = posBuf

        try allocStateBuffers()
        snapshot()
    }

    // MARK: state plumbing

    private func allocStateBuffers() throws {
        stateBindings = []
        for name in desc.stateNames {
            guard case .ndArray(let sd)? = desc.stateDescriptor(of: name) else {
                throw SpecDecodeEngine.err("state descriptor \(name)")
            }
            let shape = sd.shape.map { $0 < 0 ? kvCapacity : $0 }
            let resolved = sd.resolvingDynamicDimensions(shape)
            let byteCount = resolved.minimumByteCount
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                throw SpecDecodeEngine.err("state buffer \(name) (\(byteCount) bytes)")
            }
            memset(buffer.contents(), 0, byteCount)
            stateBindings.append(StateBinding(
                name: name, buffer: buffer, scalarType: sd.scalarType,
                shape: shape, strides: resolved.preferredStrides, byteCount: byteCount))
        }
    }

    /// Fresh stream (per chat turn): zeroed states, empty committed stream. The owned
    /// buffers are reused (memset to zero) rather than re-allocated.
    func resetStream() {
        for binding in stateBindings {
            memset(binding.buffer.contents(), 0, binding.byteCount)
        }
        stream = []
        anchor = 0
        snapshot()
    }

    // Only the cumulative GDN states need snapshotting; KV rows at positions ≥ m are
    // rewritten by every feed before they can be attended.
    private var cumulativeStates: [StateBinding] {
        stateBindings.filter { $0.name.localizedCaseInsensitiveContains("conv")
            || $0.name.localizedCaseInsensitiveContains("rec") }
    }

    private func snapshot() {
        for binding in cumulativeStates {
            var bytes = [UInt8](repeating: 0, count: binding.byteCount)
            bytes.withUnsafeMutableBytes { dst in
                memcpy(dst.baseAddress!, binding.buffer.contents(), binding.byteCount)
            }
            snapshotStates[binding.name] = bytes
        }
    }

    private func restore() {
        for binding in cumulativeStates {
            guard let saved = snapshotStates[binding.name] else { continue }
            saved.withUnsafeBytes { src in
                memcpy(binding.buffer.contents(), src.baseAddress!, binding.byteCount)
            }
        }
    }

    // MARK: forward

    private func forward(_ tokens: [Int32]) async throws {
        precondition(tokens.count == s, "window feed must be exactly S tokens")
        guard stateBindings.count == 4 else {
            throw SpecDecodeEngine.err("expected 4 hybrid states, got \(stateBindings.count)")
        }
        let positions = anchor + s

        // Write the S query tokens into the owned ids buffer (position_ids ride the pre-filled ramp).
        let idPtr = idBuffer.contents().bindMemory(to: Int32.self, capacity: s)
        for i in 0..<s { idPtr[i] = tokens[i] }

        let idShape = [1, s]
        let idStrides = try resolvedStrides(descriptor: inDesc, shape: idShape)
        let idValue = InferenceFunction.AsyncValue(
            unsafeBuffer: idBuffer, byteOffset: 0,
            scalarType: idScalar, shape: idShape, strides: idStrides)
        let posShape = [1, positions]
        let posStrides = try resolvedStrides(descriptor: posDesc, shape: posShape)
        let posValue = InferenceFunction.AsyncValue(
            unsafeBuffer: posBuffer, byteOffset: 0,
            scalarType: posScalar, shape: posShape, strides: posStrides)

        // States bound in place over the owned buffers (Core AI updates them on the GPU).
        // Distinct locals for the AsyncMutableViews lifetime rule — the view's lifetime is
        // tied to each inserted value VARIABLE, so an array-element borrow won't satisfy the
        // checker. The qwen3_5 hybrid always exposes exactly 4 (key, value, conv, rec).
        let b0 = stateBindings[0], b1 = stateBindings[1]
        let b2 = stateBindings[2], b3 = stateBindings[3]
        var s0 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: b0.buffer, byteOffset: 0, scalarType: b0.scalarType,
            shape: b0.shape, strides: b0.strides)
        var s1 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: b1.buffer, byteOffset: 0, scalarType: b1.scalarType,
            shape: b1.shape, strides: b1.strides)
        var s2 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: b2.buffer, byteOffset: 0, scalarType: b2.scalarType,
            shape: b2.shape, strides: b2.strides)
        var s3 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: b3.buffer, byteOffset: 0, scalarType: b3.scalarType,
            shape: b3.shape, strides: b3.strides)
        var stateViews = InferenceFunction.AsyncMutableViews()
        stateViews.insert(&s0, for: b0.name)
        stateViews.insert(&s1, for: b1.name)
        stateViews.insert(&s2, for: b2.name)
        stateViews.insert(&s3, for: b3.name)

        let logitsShape = [1, s, vocab]
        let logitsStrides = try resolvedStrides(descriptor: logitsDesc, shape: logitsShape)
        var logitsOut = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: logitsBuffer, byteOffset: 0, scalarType: .float16,
            shape: logitsShape, strides: logitsStrides)
        var outputViews = InferenceFunction.AsyncMutableViews()
        outputViews.insert(&logitsOut, for: logitsName)

        let t0 = benchForwards ? ContinuousClock.now : nil
        let _ = try fn.encode(
            inputs: [inName: idValue, posName: posValue],
            states: consume stateViews, outputViews: consume outputViews, to: computeStream)
        let t1 = benchForwards ? ContinuousClock.now : nil
        // Sequential loop: the next round can't propose until it reads these logits, so drain.
        await computeStream.currentWorkCompleted()
        forwards += 1
        if let t0, let t1 {
            func ms(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
                let d = b - a
                return (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000
            }
            FileHandle.standardError.write(Data(String(
                format: "FWD role=%@ pos=%d enc=%.1f drain=%.1f n=%d\n",
                role, positions, ms(t0, t1), ms(t1, .now), forwards).utf8))
        }
    }

    /// Greedy argmax over ONE logits row, read straight from the fp16 output buffer — no
    /// [1,S,vocab] flatten (that CPU pass, 2.2M elems/forward, was ~half the decode wall).
    /// Valid until the next forward overwrites the buffer, which the sequential loop honours.
    func argmax(row: Int) -> Int32 {
        let ptr = logitsBuffer.contents().bindMemory(to: Float16.self, capacity: s * vocab)
        let base = row * vocab
        var bestIndex = 0
        var bestValue: Float16 = -.infinity
        for j in 0..<vocab where ptr[base + j] > bestValue {
            bestValue = ptr[base + j]
            bestIndex = j
        }
        return Int32(bestIndex)
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

    /// Runs feed = tail+extra+pad, discarding the state advance, and returns the row base:
    /// `argmax(row: base + i)` is the prediction after extra[i-1] (row `base-1+i` overall).
    func peek(_ extra: [Int32]) async throws -> Int {
        let t = tail
        precondition(t.count + extra.count <= s, "peek overflows the window")
        let feed = t + extra + [Int32](repeating: padToken, count: s - t.count - extra.count)
        try await forward(feed)
        restore()
        return t.count
    }

    /// One verify forward: feed = tail+[a0]+drafts+pad. Returns the row base; the caller
    /// argmaxes rows base..base+drafts.count, decides acceptance, then calls commitOrRestore.
    func verifyRound(anchor a0: Int32, drafts: [Int32]) async throws -> Int {
        let t = tail
        precondition(t.count + 1 + drafts.count <= s, "verify round overflows the window")
        roundDrafts = drafts.count
        roundPad = s - t.count - 1 - drafts.count
        let feed = t + [a0] + drafts + [Int32](repeating: padToken, count: roundPad)
        try await forward(feed)
        return t.count
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
