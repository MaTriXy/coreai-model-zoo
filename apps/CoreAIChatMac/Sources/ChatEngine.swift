// ChatEngine — loads a Core AI language bundle with Apple's official runtime
// (CoreAILanguageModels) and streams chat completions with live performance
// stats. Works with any bundle exported by `coreai.llm.export` (gpt-oss,
// qwen3, gemma3, mistral, zoo models, ...).

import CoreAILanguageModels
import Darwin
import Foundation
import Tokenizers

struct ModelEntry: Identifiable, Hashable {
    let url: URL
    let sizeBytes: Int64

    var id: URL { url }
    var name: String { url.lastPathComponent }
    var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

struct ChatMessage: Identifiable, Equatable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var thinking: String = ""   // harmony "analysis" channel (gpt-oss)
    var content: String = ""
    var isStreaming = false
}

struct LiveStats: Equatable {
    var loadSeconds: Double?
    var promptTokens: Int = 0
    var ttftSeconds: Double?
    var generatedTokens: Int = 0
    var tokensPerSecond: Double?
    var footprintBytes: UInt64 = 0
    var specNote: String?       // spec-decode acceptance line (⚡Spec models only)
}

@MainActor
final class ChatEngine: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published var selectedModel: ModelEntry?
    @Published var status: Status = .idle
    @Published var messages: [ChatMessage] = []
    @Published var stats = LiveStats()

    enum Status: Equatable {
        case idle, loading, ready, generating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model"
            case .loading: return "Loading…"
            case .ready: return "Ready"
            case .generating: return "Generating…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    private var engine: (any InferenceEngine)?
    private var tokenizer: (any Tokenizer)?
    private var llada: LLaDAEngine?           // set instead of `engine` for diffusion-LM bundles
    private var spec: SpecDecodeEngine?       // set instead of `engine` for ⚡Spec verify bundles
    @Published var specOn = true              // lossless toggle: OFF = same output, no drafts
    var specLoaded: Bool { spec != nil }
    private var generationTask: Task<Void, Never>?
    private var genSeq = 0              // identifies the latest turn (stale tasks don't touch status)
    private var stopRequested = false   // user pressed Stop — stop displaying, keep draining

    // MARK: - Model discovery

    // Scan `folder` PLUS the app's own download directory, so models pulled via DownloadsView
    // (which always land in appModelsDir) are listed even when the chosen folder is a different —
    // or stale/deleted — path. Bundles found under both paths are de-duplicated.
    func scanFolder(_ folder: URL) {
        scan(folders: [Self.appModelsDir, folder])
    }

    private func scan(folders: [URL]) {
        let fm = FileManager.default
        var seen = Set<String>()
        var found: [ModelEntry] = []
        for folder in folders {
            let entries = (try? fm.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
            for url in entries
            where fm.fileExists(atPath: url.appendingPathComponent("metadata.json").path) {
                let resolved = url.resolvingSymlinksInPath()
                guard seen.insert(resolved.path).inserted else { continue }   // same bundle via two paths
                found.append(ModelEntry(url: url, sizeBytes: Self.directorySize(resolved)))
            }
        }
        models = found.sorted { $0.sizeBytes < $1.sizeBytes }
    }

    private static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    // Delete a model bundle from disk (it can always be re-downloaded). If it's the one currently
    // loaded, tear the engine down first so its memory-mapped files are released, then drop it from
    // the list. Matched by resolved path so it works whether the URL came from the sidebar entry or
    // a freshly-built download path.
    func deleteModel(at url: URL) {
        let target = url.resolvingSymlinksInPath().path
        if let sel = selectedModel, sel.url.resolvingSymlinksInPath().path == target {
            generationTask?.cancel()
            generationTask = nil
            engine = nil
            tokenizer = nil
            llada = nil
            spec = nil
            selectedModel = nil
            messages = []
            stats = LiveStats()
            status = .idle
        }
        try? FileManager.default.removeItem(at: url)
        models.removeAll { $0.url.resolvingSymlinksInPath().path == target }
    }

    // MARK: - Load

    func load(_ entry: ModelEntry) {
        generationTask?.cancel()
        selectedModel = entry
        status = .loading
        messages = []
        stats = LiveStats()
        engine = nil
        tokenizer = nil
        llada = nil
        spec = nil

        Task {
            do {
                // Diffusion-LM bundles (kind "dllm") use a host denoising loop, not the AR engine.
                if LLaDAEngine.isDLLM(bundleURL: entry.url) {
                    let start = SuspendingClock.now
                    let loaded = try await LLaDAEngine(bundleURL: entry.url)
                    self.llada = loaded
                    self.stats.loadSeconds = Self.seconds(from: start, to: .now)
                    self.stats.footprintBytes = Self.physFootprint()
                    self.status = .ready
                    return
                }

                // ⚡Spec verify bundles (static S-window graph + paired small draft) run the
                // lossless speculative-decoding loop, not the AR engine.
                if SpecDecodeEngine.isSpecVerify(bundleURL: entry.url) {
                    guard let draftURL = SpecDecodeEngine.findDraft(for: entry.url) else {
                        throw NSError(domain: "ChatEngine", code: 2, userInfo: [
                            NSLocalizedDescriptionKey:
                                "no draft bundle next to \(entry.name) — download the ⚡Spec draft too"])
                    }
                    let start = SuspendingClock.now
                    let loaded = try await SpecDecodeEngine(targetURL: entry.url, draftURL: draftURL)
                    self.spec = loaded
                    self.stats.loadSeconds = Self.seconds(from: start, to: .now)
                    self.stats.footprintBytes = Self.physFootprint()
                    self.status = .ready
                    return
                }

                let bundle = try LanguageBundle(from: entry.url.path)
                let engineConfig = ModelConfig(
                    name: bundle.name,
                    tokenizer: bundle.tokenizer,
                    vocabSize: bundle.vocabSize,
                    maxContextLength: bundle.maxContextLength,
                    serializedModel: [bundle.modelAssetPath],
                    function: bundle.language.functionMap?.name(for: "main") ?? "main"
                )
                let configData = try JSONEncoder().encode(engineConfig)
                let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

                let start = SuspendingClock.now
                // These zoo bundles are decode-pipelined (custom Metal-kernel) models. The factory's
                // auto-detect maps every "dynamic" structure to the GPU "pipelined" variant, whose
                // logits path asserts in GrowingLogitsBuffer for them (SIGTRAP on load). The
                // "coreai-sequential" variant is the one that drives these bundles correctly; it is
                // also compatible with any other dynamic bundle (chunked-static ones throw cleanly).
                async let engineResult = EngineFactory.createEngine(
                    config: configData, modelURL: modelURL,
                    options: EngineOptions(variant: "coreai-sequential"))
                async let tokenizerResult = bundle.loadTokenizer()
                let loadedEngine = try await engineResult
                let loadedTokenizer = try await tokenizerResult
                let elapsed = Self.seconds(from: start, to: .now)

                self.engine = loadedEngine
                self.tokenizer = loadedTokenizer
                self.stats.loadSeconds = elapsed
                self.stats.footprintBytes = Self.physFootprint()
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    // MARK: - Chat

    func send(_ text: String) {
        guard status == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if llada != nil { sendDLLM(trimmed); return }
        if spec != nil { sendSpec(trimmed); return }
        guard let engine, let tokenizer else { return }

        messages.append(ChatMessage(role: .user, content: trimmed))
        var reply = ChatMessage(role: .assistant, isStreaming: true)
        let replyID = reply.id
        messages.append(reply)
        status = .generating
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil
        stopRequested = false
        genSeq += 1
        let seq = genSeq

        // Full-history prompt via the bundle's own chat template (multi-turn).
        // Assistant turns feed back only the final answer (harmony convention).
        let history: [[String: any Sendable]] = messages.dropLast().map {
            ["role": $0.role == .user ? "user" : "assistant", "content": $0.content]
        }

        // The CoreAI engine `generate()` runs an UNSTRUCTURED task that produces exactly `maxTokens`
        // and can't be cancelled (no `onTermination`, no stop in `InferenceOptions`). Stop sequences
        // only halt the DISPLAY, not the engine — so after a short answer the engine keeps cranking
        // in the background. `reset()` then hard-asserts (`drain()` SIGTRAP) if called while it's
        // busy. So: (1) wait for the previous turn to fully drain before reusing the engine,
        // (2) consume this turn's stream to its end (drain) even after the visible answer stops,
        // (3) cap `maxTokens` so a drain is bounded. The user can keep typing once the answer is in;
        // the next turn just awaits the (usually finished) drain.
        let previous = generationTask
        generationTask = Task {
            await previous?.value
            do {
                let promptTokens = try tokenizer.applyChatTemplate(messages: history)
                if seq == self.genSeq { self.stats.promptTokens = promptTokens.count }
                try await engine.reset()

                let stops = StopSequences(for: tokenizer)
                let requestStart = SuspendingClock.now
                var firstTokenAt: SuspendingClock.Instant?
                var genTokens: [Int] = []       // for tokenizer.decode ([Int])
                var recent: [Int32] = []         // for StopSequences.matches ([Int32])
                var displaying = true        // false after a stop sequence / user Stop — keep draining
                var emitted = 0

                for try await output in try engine.generate(
                    with: promptTokens.map(Int32.init),
                    samplingConfiguration: SamplingConfiguration(temperature: 0.7),
                    inferenceOptions: InferenceOptions(maxTokens: 256, includeLogits: false)
                ) {
                    guard displaying else { continue }   // engine still producing — drain, don't show
                    if self.stopRequested { displaying = false; self.finalize(&reply, replyID, seq); continue }
                    if firstTokenAt == nil {
                        firstTokenAt = .now
                        if seq == self.genSeq { self.stats.ttftSeconds = Self.seconds(from: requestStart, to: firstTokenAt!) }
                    }
                    recent.append(output.tokenId)
                    if recent.count > stops.maxLength { recent.removeFirst(recent.count - stops.maxLength) }
                    if stops.matches(recentTokens: recent) { displaying = false; self.finalize(&reply, replyID, seq); continue }

                    genTokens.append(Int(output.tokenId))
                    emitted += 1
                    let parsed = HarmonyParser.parse(tokenizer.decode(tokens: genTokens))
                    reply.thinking = parsed.thinking
                    reply.content = parsed.answer
                    self.update(replyID, with: reply)
                    if seq == self.genSeq {
                        self.stats.generatedTokens = emitted
                        if let first = firstTokenAt, emitted > 1 {
                            let genElapsed = Self.seconds(from: first, to: .now)
                            if genElapsed > 0 { self.stats.tokensPerSecond = Double(emitted - 1) / genElapsed }
                        }
                    }
                }
                if displaying { self.finalize(&reply, replyID, seq) }   // hit maxTokens without a stop
            } catch {
                reply.isStreaming = false
                if reply.content.isEmpty { reply.content = "(generation failed: \(error))" }
                self.update(replyID, with: reply)
                if seq == self.genSeq { self.status = .ready }
            }
        }
    }

    // Mark the visible reply done. Status/footprint are only touched by the LATEST turn, so a still-
    // draining older turn can't stomp a newer turn's `.generating`.
    private func finalize(_ reply: inout ChatMessage, _ id: UUID, _ seq: Int) {
        reply.isStreaming = false
        update(id, with: reply)
        if seq == genSeq {
            status = .ready
            stats.footprintBytes = Self.physFootprint()
        }
    }

    // MARK: - Diffusion-LM chat (LLaDA / d3LLM)

    // A diffusion LM denoises the whole canvas in parallel — there is no AR token stream. We run the
    // host loop and stream the decoded canvas after each denoising step (the masks fill in). Unlike
    // the AR engine, this loop completes promptly (distillation keeps the step count low) and needs no
    // drain hack — but we still serialize turns through `generationTask`.
    private func sendDLLM(_ trimmed: String) {
        guard let llada else { return }
        messages.append(ChatMessage(role: .user, content: trimmed))
        let reply = ChatMessage(role: .assistant, isStreaming: true)
        let replyID = reply.id
        messages.append(reply)
        status = .generating
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil
        stopRequested = false
        genSeq += 1
        let seq = genSeq

        let history: [[String: any Sendable]] = messages.dropLast().map {
            ["role": $0.role == .user ? "user" : "assistant", "content": $0.content]
        }

        let previous = generationTask
        generationTask = Task {
            await previous?.value
            do {
                if let ids = try? llada.promptIds(history: history), seq == self.genSeq {
                    self.stats.promptTokens = ids.count
                }
                let requestStart = SuspendingClock.now
                let result = try await llada.generate(history: history) { info in
                    guard !self.stopRequested else { return }
                    if seq == self.genSeq, self.stats.ttftSeconds == nil {
                        self.stats.ttftSeconds = Self.seconds(from: requestStart, to: .now)
                    }
                    self.setContent(replyID, info.text)
                    if seq == self.genSeq {
                        self.stats.generatedTokens = info.committed
                        let elapsed = Self.seconds(from: requestStart, to: .now)
                        if elapsed > 0 { self.stats.tokensPerSecond = Double(info.committed) / elapsed }
                    }
                }
                if !self.stopRequested { self.setContent(replyID, result.text) }
                if seq == self.genSeq { self.stats.generatedTokens = result.committed }
                self.finalizeByID(replyID, seq)
            } catch {
                self.setContent(replyID, "(generation failed: \(error))")
                self.finalizeByID(replyID, seq)
            }
        }
    }

    // MARK: - Speculative-decoding chat (⚡Spec verify bundles)

    // Greedy two-model spec-decode: the loop is synchronous per round (draft → one verify
    // forward → commit), streams the committed text, and finishes promptly — no drain hack.
    // `specOn` is read per turn, so the toggle A/Bs lossless speed on the same conversation.
    private func sendSpec(_ trimmed: String) {
        guard let spec else { return }
        messages.append(ChatMessage(role: .user, content: trimmed))
        let reply = ChatMessage(role: .assistant, isStreaming: true)
        let replyID = reply.id
        messages.append(reply)
        status = .generating
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil
        stats.specNote = nil
        stopRequested = false
        genSeq += 1
        let seq = genSeq
        let useSpec = specOn

        let history: [[String: any Sendable]] = messages.dropLast().map {
            ["role": $0.role == .user ? "user" : "assistant", "content": $0.content]
        }

        let previous = generationTask
        generationTask = Task {
            await previous?.value
            do {
                let requestStart = SuspendingClock.now
                let result = try await spec.generate(history: history, specOn: useSpec) { text, st in
                    guard !self.stopRequested else { return }
                    if seq == self.genSeq, self.stats.ttftSeconds == nil {
                        self.stats.ttftSeconds = Self.seconds(from: requestStart, to: .now)
                        self.stats.promptTokens = st.promptTokens
                    }
                    self.setContent(replyID, text)
                    if seq == self.genSeq {
                        self.stats.generatedTokens = st.generated
                        self.stats.specNote = useSpec ? st.note : nil
                        let elapsed = Self.seconds(from: requestStart, to: .now)
                        if elapsed > 0 { self.stats.tokensPerSecond = Double(st.generated) / elapsed }
                    }
                }
                if !self.stopRequested { self.setContent(replyID, result.text) }
                if seq == self.genSeq {
                    self.stats.generatedTokens = result.stats.generated
                    self.stats.specNote = useSpec ? result.stats.note : nil
                }
                self.finalizeByID(replyID, seq)
            } catch {
                self.setContent(replyID, "(generation failed: \(error))")
                self.finalizeByID(replyID, seq)
            }
        }
    }

    private func setContent(_ id: UUID, _ text: String) {
        if let i = messages.firstIndex(where: { $0.id == id }) { messages[i].content = text }
    }

    private func finalizeByID(_ id: UUID, _ seq: Int) {
        if let i = messages.firstIndex(where: { $0.id == id }) { messages[i].isStreaming = false }
        if seq == genSeq {
            status = .ready
            stats.footprintBytes = Self.physFootprint()
        }
    }

    func stopGeneration() {
        // The engine can't be interrupted mid-generation; stop showing tokens. It keeps draining in
        // the background and the next turn waits for it.
        stopRequested = true
    }

    private func update(_ id: UUID, with message: ChatMessage) {
        if let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index] = message
        }
    }

    // MARK: - Helpers

    static func seconds(from start: SuspendingClock.Instant, to end: SuspendingClock.Instant) -> Double {
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }
}

// Splits gpt-oss "harmony" output into the analysis channel (thinking) and the
// final channel (answer). Models without harmony markers pass through as-is.
enum HarmonyParser {
    static func parse(_ raw: String) -> (thinking: String, answer: String) {
        guard raw.contains("<|channel|>") else {
            return ("", strip(raw))
        }
        var thinking = ""
        var answer = ""
        if let analysisRange = raw.range(of: "<|channel|>analysis<|message|>") {
            let afterAnalysis = raw[analysisRange.upperBound...]
            if let end = afterAnalysis.range(of: "<|end|>") {
                thinking = String(afterAnalysis[..<end.lowerBound])
            } else {
                thinking = String(afterAnalysis)
            }
        }
        if let finalRange = raw.range(of: "<|channel|>final<|message|>") {
            answer = String(raw[finalRange.upperBound...])
        }
        return (strip(thinking), strip(answer))
    }

    private static func strip(_ text: String) -> String {
        var out = text
        for marker in ["<|return|>", "<|end|>", "<|endoftext|>"] {
            out = out.replacingOccurrences(of: marker, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
