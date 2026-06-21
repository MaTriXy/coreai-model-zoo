// AudioModel — view model for coreai-audio. Loads the Qwen2.5-Omni audio-understanding model via
// CoreAIKit's KitAudioModel (downloads from Hugging Face on first run), attaches a recorded/chosen/
// demo clip, and answers "what do you hear?" through a plain FoundationModels LanguageModelSession.
//
// Built on the CoreAIKit *library* (not raw coreai-models products): the library packages the
// swift-transformers / swift-crypto resource bundles so they code-sign cleanly on device.

import CoreAIKit
import Foundation
import FoundationModels

@MainActor
final class AudioModel: ObservableObject {
    @Published var status = "Tap “Load model” (first run downloads ~5 GB)."
    @Published var loaded = false
    @Published var busy = false
    @Published var recording = false
    @Published var clipName = "No audio loaded."
    @Published var answer = ""

    private var model: KitAudioModel?
    private var samples: [Float]?
    private let recorder = MicRecorder()

    /// Clip cap (s): the decoder prefills audio tokens one step at a time on device, so a long clip
    /// is a slow prefill (the proven demo is 4 s ≈ 100 tokens). Cap for snappy responses.
    private let maxClipSeconds = 6

    /// HF bundles: iOS pulls the AOT `.aimodelc` decoder (the 3.9 GB JIT graph jetsams on iPhone);
    /// macOS pulls the GPU-pipelined JIT decoder. Encoder is shared.
    private var modelID: AudioModelID {
        let repo = "mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI"
        #if os(iOS)
            let decoderPath = "ios/qwen2_5_omni_3b_thinker_n750_ios"
        #else
            let decoderPath = "gpu-pipelined/qwen2_5_omni_3b_thinker_int8lin_n750_s1"
        #endif
        return AudioModelID(
            decoder: ModelID(repo, path: decoderPath),
            encoder: ModelID(repo, path: "gpu-pipelined/qwen2_5_omni_3b_audio_encoder_fp16_k15"),
            arch: .qwen2_5Omni3B)
    }

    func load() async {
        guard !loaded, !busy else { return }
        busy = true
        status = "Downloading + loading model…"
        do {
            let m = try await KitAudioModel(model: modelID) { progress in
                Task { @MainActor [weak self] in
                    self?.status = String(
                        format: "Downloading %@ — %.0f%%",
                        (progress.currentFile as NSString).lastPathComponent, progress.fraction * 100)
                }
            }
            self.model = m
            loaded = true
            status = "Model ready. Record / Choose / Demo, then Ask."
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        busy = false
    }

    func loadFile(_ url: URL) {
        guard let pcm = AudioLoader.load16kMono(url) else {
            status = "Could not decode \(url.lastPathComponent)."
            return
        }
        samples = pcm
        answer = ""
        clipName = "\(url.lastPathComponent)  (\(String(format: "%.1f", Double(pcm.count) / 16000))s)"
        status = "Audio loaded. Ask a question."
    }

    func loadDemoNoise() {
        samples = AudioLoader.demoNoise()
        answer = ""
        clipName = "Demo: white noise (4s)"
        status = "Demo clip loaded. Ask a question."
    }

    /// Toggle mic recording. On stop, the captured clip becomes the audio to ask about.
    func toggleRecord() {
        if recording {
            recording = false
            status = "Processing recording…"
            recorder.stop { [weak self] pcm in
                Task { @MainActor in
                    guard let self else { return }
                    self.samples = pcm
                    self.clipName = String(format: "Mic clip (%.1fs)", Double(pcm.count) / 16000)
                    self.status = pcm.isEmpty ? "No audio captured." : "Recorded. Ask a question."
                }
            }
        } else {
            recording = true
            answer = ""
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

    func ask(_ question: String) async {
        guard let model, let samples, !busy else { return }
        busy = true
        answer = ""
        let clip = Array(samples.prefix(16000 * maxClipSeconds))
        status = "Encoding audio…"
        do {
            try await model.attach(samples: clip)  // mel → encoder → static buffer
            status = "Thinking… (on-device, a few seconds)"
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(
                to: question.isEmpty ? "What do you hear?" : question,
                options: GenerationOptions(maximumResponseTokens: 128))
            for try await snapshot in stream { answer = snapshot.content }
            status = "Done."
        } catch {
            status = "Generation failed: \(error.localizedDescription)"
        }
        busy = false
    }
}
