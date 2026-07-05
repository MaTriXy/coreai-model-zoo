// TranscribeModel — view model for the Transcribe tab. Turns a recorded/chosen/demo clip into TEXT
// (speech-to-text) with a CHOICE of four Core AI ASR models, distinct from the Understand tab (which
// describes sounds):
//   • Whisper large-v3-turbo (Apple-recipe export on the stock runtime, via CoreAIKit's
//     KitWhisperModel) — published, downloads from the Hub on both platforms.
//   • Qwen3-ASR-1.7B (the zoo's first ASR, via CoreAIKit's KitASRModel).
//   • Parakeet-TDT-0.6B (the zoo's first transducer / TDT, via CoreAIKit's KitParakeetModel).
//   • Nemotron 3.5 Streaming 0.6B (the zoo's first STREAMING ASR, via CoreAIKit's
//     KitNemotronModel) — adds a LIVE mode: continuous mic transcription in 320 ms chunks,
//     any-length audio, 40 locales, punctuation built in.
//
// All models download from the Hugging Face Hub on first load (iOS + macOS) and cache on-device.

import CoreAIKit
import Foundation

@MainActor
final class TranscribeModel: ObservableObject {
    /// The selectable transcription engines.
    enum Engine: String, CaseIterable, Identifiable {
        case whisper, qwen3ASR, parakeet, nemotron
        var id: String { rawValue }
        var title: String {
            switch self {
            case .whisper: return "Whisper large-v3-turbo"
            case .qwen3ASR: return "Qwen3-ASR-1.7B"
            case .parakeet: return "Parakeet TDT 0.6B"
            case .nemotron: return "Nemotron Streaming 0.6B"
            }
        }
        var blurb: String {
            switch self {
            case .whisper: return "OpenAI Whisper on Core AI (stock runtime). 100 languages, ≤30 s."
            case .qwen3ASR: return "The zoo's first ASR. 52 languages, ≤30 s."
            case .parakeet: return "NVIDIA FastConformer transducer — the zoo's first TDT/RNN-T. 25 EU languages, ≤30 s."
            case .nemotron: return "The zoo's first STREAMING ASR — live mic, 320 ms chunks, 40 locales, any length."
            }
        }
    }

    @Published var engine: Engine = .whisper { didSet { if oldValue != engine { unload() } } }
    @Published var status = "Pick an engine, then tap “Load model”."
    @Published var loaded = false
    @Published var busy = false
    @Published var recording = false
    @Published var live = false
    @Published var clipName = "No audio loaded."
    @Published var transcript = ""
    @Published var language = ""

    private var whisper: KitWhisperModel?
    private var asr: KitASRModel?
    private var parakeet: KitParakeetModel?
    private var nemotron: KitNemotronModel?
    private var samples: [Float]?
    private let recorder = MicRecorder()
    private let micStreamer = MicStreamer()
    private var liveTask: Task<Void, Never>?

    /// The batch bundles transcribe ≤30 s clips; Nemotron streams any length.
    private let maxClipSeconds = 30

    // MARK: - Loading

    func load() async {
        guard !loaded, !busy else { return }
        busy = true
        status = "Loading \(engine.title)…"
        do {
            switch engine {
            case .whisper:
                whisper = try await KitWhisperModel(model: .largeV3Turbo) { progress in
                    Task { @MainActor [weak self] in
                        self?.status = String(
                            format: "Downloading %@ — %.0f%%",
                            (progress.currentFile as NSString).lastPathComponent,
                            progress.fraction * 100)
                    }
                }
            case .qwen3ASR:
                asr = try await KitASRModel(model: .qwen3ASR1_7B) { progress in
                    Task { @MainActor [weak self] in
                        self?.status = String(
                            format: "Downloading %@ — %.0f%%",
                            (progress.currentFile as NSString).lastPathComponent,
                            progress.fraction * 100)
                    }
                }
            case .parakeet:
                // Prefer a sideloaded bundle (AOT-compiled encoder) — the 1.2 GB encoder's on-device
                // JIT specialization stalls, so on iPhone we ship a precompiled `.aimodelc`.
                if let local = Self.sideloadedBundle(named: "Parakeet") {
                    status = "Loading Parakeet (sideloaded)…"
                    parakeet = try await KitParakeetModel(bundleAt: local)
                } else {
                    parakeet = try await KitParakeetModel(model: .parakeetTDT) { progress in
                        Task { @MainActor [weak self] in
                            self?.status = String(
                                format: "Downloading %@ — %.0f%%",
                                (progress.currentFile as NSString).lastPathComponent,
                                progress.fraction * 100)
                        }
                    }
                }
            case .nemotron:
                // Same AOT story as Parakeet: the 1.2 GB conformer graph ships precompiled for iPhone.
                if let local = Self.sideloadedBundle(named: "Nemotron") {
                    status = "Loading Nemotron (sideloaded)…"
                    nemotron = try await KitNemotronModel(bundleAt: local)
                } else {
                    nemotron = try await KitNemotronModel(model: .nemotronASRStreaming) { progress in
                        Task { @MainActor [weak self] in
                            self?.status = String(
                                format: "Downloading %@ — %.0f%%",
                                (progress.currentFile as NSString).lastPathComponent,
                                progress.fraction * 100)
                        }
                    }
                }
            }
            loaded = true
            status = "Model ready. Record / Choose / Demo, then Transcribe."
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        busy = false
    }

    /// A sideloaded bundle in `Documents/Models/<name>` — the AOT-encoder path for iPhone, where
    /// the big encoder graphs' on-device JIT specialization stalls. Returns the directory if it
    /// holds any graph (`.aimodelc` AOT or `.aimodel`), else nil (→ Hub download).
    private static func sideloadedBundle(named name: String) -> URL? {
        let fm = FileManager.default
        let dir = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appending(path: "Models/\(name)")
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return nil }
        let hasGraph = entries.contains {
            $0.pathExtension == "aimodelc" || $0.pathExtension == "aimodel"
        }
        return hasGraph ? dir : nil
    }

    /// Drop the loaded model(s) — called when the engine selection changes.
    private func unload() {
        stopLive()
        whisper = nil
        asr = nil
        parakeet = nil
        nemotron = nil
        loaded = false
        transcript = ""; language = ""
        status = "Engine: \(engine.title). Tap “Load model”."
    }

    func loadFile(_ url: URL) {
        guard let pcm = AudioLoader.load16kMono(url) else {
            status = "Could not decode \(url.lastPathComponent)."
            return
        }
        samples = pcm
        transcript = ""; language = ""
        clipName = "\(url.lastPathComponent)  (\(String(format: "%.1f", Double(pcm.count) / 16000))s)"
        status = "Audio loaded. Tap Transcribe."
    }

    func toggleRecord() {
        if recording {
            recording = false
            status = "Processing recording…"
            recorder.stop { [weak self] pcm in
                Task { @MainActor in
                    guard let self else { return }
                    self.samples = pcm
                    self.clipName = String(format: "Mic clip (%.1fs)", Double(pcm.count) / 16000)
                    self.status = pcm.isEmpty ? "No audio captured." : "Recorded. Tap Transcribe."
                }
            }
        } else {
            recording = true
            transcript = ""; language = ""
            clipName = "Recording… tap Stop when done."
            status = "Listening…"
            recorder.start { [weak self] error in
                Task { @MainActor in
                    guard let self, let error else { return }
                    self.recording = false
                    self.clipName = "No audio loaded."
                    self.status = "Mic error: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Live streaming (Nemotron)

    /// Toggle continuous live-mic transcription — the Nemotron differentiator: 320 ms chunks
    /// stream through the cache-aware encoder while you speak; the transcript grows in place.
    func toggleLive() {
        if live { stopLive(); return }
        guard engine == .nemotron, let nemotron, !busy, !recording else { return }
        live = true
        transcript = ""; language = ""
        clipName = "Live — speak; tap Stop when done."
        status = "Listening… (streaming on-device)"
        liveTask = Task { [weak self] in
            guard let self else { return }
            do {
                let session = try nemotron.makeSession(language: "en-US")
                let stream = try await self.micStreamer.start()
                var seconds = 0.0
                for await packet in stream {
                    let partial = try await session.feed(samples: packet)
                    seconds += Double(packet.count) / 16000
                    let s = seconds
                    await MainActor.run {
                        if !partial.isEmpty { self.transcript = partial }
                        self.clipName = String(format: "Live — %.0fs. Tap Stop when done.", s)
                    }
                }
                let result = try await session.finish()
                let avgMs = session.totalChunkSeconds / Double(max(session.chunkCount, 1)) * 1000
                await MainActor.run {
                    if !result.text.isEmpty { self.transcript = result.text }
                    self.language = result.language
                    self.status = String(
                        format: "Live done — %d chunks, avg %.0f ms/chunk (320 ms audio).",
                        session.chunkCount, avgMs)
                }
            } catch {
                await MainActor.run {
                    self.status = "Live failed: \(error.localizedDescription)"
                    self.live = false
                }
            }
        }
    }

    private func stopLive() {
        guard live else { return }
        live = false
        clipName = "No audio loaded."
        status = "Finishing live transcript…"
        micStreamer.stop()   // ends the packet stream; the live task then finish()es the session
    }

    // MARK: - Transcription

    func transcribeClip() async {
        guard loaded, let samples, !busy, !live else { return }
        busy = true
        transcript = ""; language = ""
        // Nemotron streams — no bucket, feed the whole clip; the batch engines cap at 30 s.
        let clip = engine == .nemotron ? samples : Array(samples.prefix(16000 * maxClipSeconds))
        status = "Transcribing… (on-device)"
        // Stream the running transcript into the UI as the decoder emits it. The callback fires off
        // the main actor, so hop back; the length guard keeps a late-arriving partial from
        // overwriting the final text with an earlier (shorter) snapshot.
        let onPartial: @Sendable (String) -> Void = { [weak self] partial in
            Task { @MainActor in
                guard let self else { return }
                if partial.count >= self.transcript.count { self.transcript = partial }
            }
        }
        do {
            let result: Transcription
            switch engine {
            case .whisper:
                guard let whisper else { busy = false; return }
                result = try await whisper.transcribe(samples: clip, onPartial: onPartial)
            case .qwen3ASR:
                guard let asr else { busy = false; return }
                status = "Encoding audio…"
                try await asr.attach(samples: clip)  // mel → AuT encoder → static buffer
                status = "Transcribing… (on-device)"
                result = try await asr.transcribe(onPartial: onPartial)
            case .parakeet:
                guard let parakeet else { busy = false; return }
                result = try await parakeet.transcribe(samples: clip, onPartial: onPartial)
            case .nemotron:
                guard let nemotron else { busy = false; return }
                result = try await nemotron.transcribe(samples: clip, onPartial: onPartial)
            }
            transcript = result.text
            language = result.language
            status = "Done."
        } catch {
            status = "Transcription failed: \(error.localizedDescription)"
        }
        busy = false
    }
}
