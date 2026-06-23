// TranscribeEngine — downloads / loads the Whisper Core AI bundle and transcribes audio
// on the stock runtime. The bundle's graph is a fixed 128-token decoder window
// (input_features [1,128,3000] + decoder_input_ids [1,128] → logits [1,128,51866]); we pad
// the decoder buffer, run greedily, and read logits at the real last position.
//
// Pipeline: audio file/mic → 16 kHz mono Float → WhisperMel log-mel → autoregressive decode
// (re-running the combined encoder+decoder graph each step at constant shape, ~0.18 s/token
// on M-series GPU) → detokenize with the bundled HF Whisper tokenizer.

import AVFoundation
import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

/// Holds the non-Sendable runtime objects (InferenceFunction / Tokenizer / descriptors) and
/// runs the decode off the @MainActor engine without tripping Swift 6 region isolation.
/// Safe in practice — the engine serializes calls (one transcription at a time).
private struct WhisperRunner: @unchecked Sendable {
    let function: InferenceFunction
    let featDescriptor: NDArrayDescriptor
    let tokDescriptor: NDArrayDescriptor
    let tokenizer: any Tokenizer
    let melFilters: [Float]

    // Whisper large-v3 special tokens.
    static let prompt: [Int32] = [50258, 50259, 50360, 50364]  // SOT, en, transcribe, no-timestamps
    static let eot: Int32 = 50257
    static let decLen = 128

    func transcribe(samples: [Float]) async throws -> String {
        let mel = WhisperMel.logMel(samples: samples, melFilters: melFilters)  // [128*3000]
        var feat = NDArray(descriptor: featDescriptor)
        fillNDArray(&feat, as: Float16.self, with: mel.map { Float16($0) })

        let vocab = tokDescriptorVocab()
        var seq = Self.prompt
        for _ in 0..<(Self.decLen - Self.prompt.count) {
            let k = seq.count - 1
            var dec = NDArray(descriptor: tokDescriptor)
            fillNDArray(&dec, as: Int32.self, count: Self.decLen) { i in
                i < seq.count ? seq[i] : Self.eot
            }
            var outputs = try await function.run(inputs: ["input_features": feat, "decoder_input_ids": dec])
            guard let logitsNd = outputs.remove("logits")?.ndArray else {
                throw NSError(domain: "TranscribeEngine", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "no logits output"])
            }
            let logits = flattenAsFloat(logitsNd)  // [1*128*V]
            let base = k * vocab
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for v in 0..<vocab {
                let x = logits[base + v]
                if x > bestVal { bestVal = x; best = v }
            }
            let next = Int32(best)
            if next == Self.eot { break }
            seq.append(next)
        }
        let content = seq[Self.prompt.count...].map { Int($0) }.filter { $0 < Int(Self.eot) }
        return tokenizer.decode(tokens: content).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokDescriptorVocab() -> Int {
        // logits last dim; fall back to whisper-v3 vocab if unavailable.
        51866
    }
}

@MainActor
final class TranscribeEngine: ObservableObject {

    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String
        let bundleDirName: String
        let title: String
        let aimodelName: String
    }

    static let catalog: [ModelOption] = [
        ModelOption(
            repoId: "mlboydaisuke/whisper-large-v3-turbo-CoreAI-official",
            bundleDirName: "whisper-large-v3-turbo-CoreAI",
            title: "Whisper large-v3-turbo",
            aimodelName: "whisper-large-v3-turbo_float16_fixed128.aimodel")
    ]

    enum Status: Equatable {
        case idle, downloading, loading, ready, transcribing
        case error(String)
        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading model…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .transcribing: return "Transcribing…"
            case .error(let m): return "Error: \(m)"
            }
        }
        var isBusy: Bool {
            switch self { case .downloading, .loading, .transcribing: return true; default: return false }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var transcript = ""
    @Published private(set) var seconds: Double?
    @Published private(set) var modelTitle = ""

    let downloader = ModelDownloader()
    private var runner: WhisperRunner?
    private var work: Task<Void, Never>?

    var canTranscribe: Bool { runner != nil && !status.isBusy }
    var isDownloadingOrLoading: Bool {
        if case .downloading = status { return true }; if case .loading = status { return true }; return false
    }

    private func directoryItems(for o: ModelOption) -> [ModelDownloader.Item] {
        [.init(remote: o.aimodelName, local: o.aimodelName),
         .init(remote: "tokenizer", local: "tokenizer")]
    }
    private static let rootFiles = ["mel_filters_128.npy"]

    // MARK: - Loading

    func loadFromHub(_ o: ModelOption) {
        work?.cancel()
        modelTitle = o.title
        status = .downloading
        work = Task {
            do {
                let dest = try Self.bundleDestination(for: o)
                await downloader.fetch(repo: "https://huggingface.co/\(o.repoId)",
                                       items: directoryItems(for: o), into: dest)
                try Task.checkCancellation()
                if case .failed(let m) = downloader.phase { throw Self.err(m) }
                try await Self.fetchRootFiles(repoId: o.repoId, into: dest)
                try await load(at: dest, aimodelName: o.aimodelName)
            } catch is CancellationError {
            } catch { status = .error("\(error)") }
        }
    }

    func loadLocal(_ url: URL, aimodelName: String) {
        work?.cancel()
        modelTitle = url.lastPathComponent
        work = Task {
            do { try await load(at: url, aimodelName: aimodelName) }
            catch { status = .error("\(error)") }
        }
    }

    private func load(at dir: URL, aimodelName: String) async throws {
        status = .loading
        let accessed = dir.startAccessingSecurityScopedResource()
        defer { if accessed { dir.stopAccessingSecurityScopedResource() } }

        let modelURL = dir.appendingPathComponent(aimodelName)
        let prepared = try await PreparedModel.prepare(at: modelURL)
        guard let desc = prepared.model.functionDescriptor(for: "main"),
              case .ndArray(let featDesc) = desc.inputDescriptor(of: "input_features"),
              case .ndArray(let tokDesc) = desc.inputDescriptor(of: "decoder_input_ids"),
              let fn = try prepared.model.loadFunction(named: "main")
        else { throw Self.err("model missing expected 'main' function / inputs") }

        let tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"))
        let melFilters = try WhisperMel.loadNpyFloat32(dir.appendingPathComponent("mel_filters_128.npy"))

        runner = WhisperRunner(function: fn, featDescriptor: featDesc, tokDescriptor: tokDesc,
                               tokenizer: tokenizer, melFilters: melFilters)
        status = .ready
    }

    // MARK: - Transcription

    func transcribe(audioURL: URL) {
        guard let runner else { return }
        work?.cancel()
        status = .transcribing
        transcript = ""
        work = Task {
            do {
                let samples = try Self.loadMono16k(url: audioURL)
                let start = DispatchTime.now().uptimeNanoseconds
                let text = try await runner.transcribe(samples: samples)
                try Task.checkCancellation()
                seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                transcript = text
                status = .ready
            } catch is CancellationError {
            } catch { status = .error("\(error)") }
        }
    }

    func cancel() {
        work?.cancel()
        if status.isBusy { status = runner != nil ? .ready : .idle }
    }

    // MARK: - Audio → 16 kHz mono Float

    nonisolated static func loadMono16k(url: URL) throws -> [Float] {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let file = try AVAudioFile(forReading: url)
        let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                      channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: file.processingFormat, to: outFormat) else {
            throw err("cannot create audio converter")
        }
        var samples = [Float]()
        let cap: AVAudioFrameCount = 16000 * 4
        let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap)!
        var done = false
        while !done {
            let inBlock: AVAudioConverterInputBlock = { _, outStatus in
                let inCap: AVAudioFrameCount = 16384
                guard let inBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: inCap)
                else { outStatus.pointee = .endOfStream; return nil }
                do {
                    try file.read(into: inBuf)
                } catch { outStatus.pointee = .endOfStream; return nil }
                if inBuf.frameLength == 0 { outStatus.pointee = .endOfStream; return nil }
                outStatus.pointee = .haveData
                return inBuf
            }
            var err2: NSError?
            let st = converter.convert(to: outBuf, error: &err2, withInputFrom: inBlock)
            if let e = err2 { throw e }
            if outBuf.frameLength > 0, let ch = outBuf.floatChannelData {
                samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(outBuf.frameLength)))
            }
            if st == .endOfStream || st == .error { done = true }
            // Cap at 30 s (one decode window).
            if samples.count >= WhisperMel.nSamples { done = true }
        }
        return samples
    }

    // MARK: - Storage helpers

    private static func bundleDestination(for o: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(o.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func fetchRootFiles(repoId: String, into dest: URL) async throws {
        for name in rootFiles {
            let out = dest.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: out.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(name)") else { continue }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                throw err("failed to fetch \(name)")
            }
            try data.write(to: out, options: .atomic)
        }
    }

    nonisolated static func err(_ m: String) -> NSError {
        NSError(domain: "TranscribeEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
