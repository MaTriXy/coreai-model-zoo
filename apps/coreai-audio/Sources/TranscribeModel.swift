// TranscribeModel — view model for the Transcribe tab. Turns a recorded/chosen/demo clip into TEXT
// (speech-to-text) with a CHOICE of two Core AI ASR models, distinct from the Understand tab (which
// describes sounds):
//   • Whisper large-v3-turbo (Apple-recipe export on the stock runtime, via CoreAIKit's
//     KitWhisperModel) — published, downloads from the Hub on both platforms.
//   • Qwen3-ASR-1.7B (the zoo's first ASR, via CoreAIKit's KitASRModel).
//
// Both models download from the Hugging Face Hub on first load (iOS + macOS) and cache on-device.

import CoreAIKit
import Foundation

@MainActor
final class TranscribeModel: ObservableObject {
    /// The selectable transcription engines.
    enum Engine: String, CaseIterable, Identifiable {
        case whisper, qwen3ASR, parakeet
        var id: String { rawValue }
        var title: String {
            switch self {
            case .whisper: return "Whisper large-v3-turbo"
            case .qwen3ASR: return "Qwen3-ASR-1.7B"
            case .parakeet: return "Parakeet TDT 0.6B"
            }
        }
        var blurb: String {
            switch self {
            case .whisper: return "OpenAI Whisper on Core AI (stock runtime). 100 languages, ≤30 s."
            case .qwen3ASR: return "The zoo's first ASR. 52 languages, ≤30 s."
            case .parakeet: return "NVIDIA FastConformer transducer — the zoo's first TDT/RNN-T. 25 EU languages, ≤30 s."
            }
        }
    }

    @Published var engine: Engine = .whisper { didSet { if oldValue != engine { unload() } } }
    @Published var status = "Pick an engine, then tap “Load model”."
    @Published var loaded = false
    @Published var busy = false
    @Published var recording = false
    @Published var clipName = "No audio loaded."
    @Published var transcript = ""
    @Published var language = ""

    private var whisper: KitWhisperModel?
    private var asr: KitASRModel?
    private var parakeet: KitParakeetModel?
    private var samples: [Float]?
    private let recorder = MicRecorder()

    /// Both bundles transcribe ≤30 s clips; cap to the encoder/decoder window.
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
                parakeet = try await KitParakeetModel(model: .parakeetTDT) { progress in
                    Task { @MainActor [weak self] in
                        self?.status = String(
                            format: "Downloading %@ — %.0f%%",
                            (progress.currentFile as NSString).lastPathComponent,
                            progress.fraction * 100)
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

    /// Drop the loaded model(s) — called when the engine selection changes.
    private func unload() {
        whisper = nil
        asr = nil
        parakeet = nil
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

    // MARK: - Transcription

    func transcribeClip() async {
        guard loaded, let samples, !busy else { return }
        busy = true
        transcript = ""; language = ""
        let clip = Array(samples.prefix(16000 * maxClipSeconds))
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
