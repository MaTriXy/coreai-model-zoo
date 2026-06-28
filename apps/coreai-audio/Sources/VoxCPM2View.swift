// VoxCPM2View — the "Voice 2B" tab: VoxCPM2 (2B, 48 kHz) text-to-speech on Core AI, the scaled
// successor to the VoxCPM-0.5B "Voice" tab. Drives CoreAIKit's `VoxCPM2TTS` (five bundles + host glue,
// all on device). Kept separate from VoxCPMView so the shipped 0.5B path is untouched.
//
// Assets load from the same `VoxCPMAssets` root (symlink -> conversion artifacts) — the v2 bundles are
// `voxcpm2_*`, glue is `voxcpm2_host_glue/`, tokenizer is `tokenizer2/` (LlamaTokenizer fast path).
import CoreAIKit
import SwiftUI

enum VoxCPM2Assets {
    static let repo = "mlboydaisuke/VoxCPM2-CoreAI"   // not yet published

    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("VoxCPMAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VoxCPMAssets")
        #endif
    }

    /// Present only if the v2 tokenizer is staged (distinguishes from the v1-only asset root).
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("tokenizer2").path) ? p : nil
    }

    static func paths(root: URL, lm: VoxCPM2Paths.LMPrecision = .int8) -> VoxCPM2Paths {
        let tok = root.appendingPathComponent("tokenizer2")
        #if os(macOS)
        return .standard(artifactsRoot: root, lm: lm, tokenizerDir: tok)
        #else
        return .aot(root: root, arch: "h18p", lm: lm, tokenizerDir: tok)
        #endif
    }
}

@MainActor
final class VoxCPM2VM: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var streaming = true
    @Published var lm: VoxCPM2Paths.LMPrecision = .int8

    private var tts: VoxCPM2TTS?
    private let player = AudioPlayer()
    private var preroll: [[Float]] = []
    private var prerolling = false
    private let prerollChunks = 3                       // one chunk ≈ 0.16 s -> ~0.48 s lead

    private var sr: Double { Double(VoxCPM2TTS.sampleRate) }   // 48 kHz

    func load() async {
        busy = true; status = "Loading VoxCPM2 2B (\(lm == .fp16 ? "fp16" : "int8"))…"
        do {
            guard let root = VoxCPM2Assets.root else {
                status = "v2 model not found (need tokenizer2/ + voxcpm2_* at \(VoxCPM2Assets.location.path))."
                busy = false; return
            }
            let t0 = Date()
            tts = try await VoxCPM2TTS(paths: VoxCPM2Assets.paths(root: root, lm: lm))
            loaded = true
            status = String(format: "Ready (%@, loaded in %.1f s).", lm == .fp16 ? "fp16" : "int8",
                            Date().timeIntervalSince(t0))
        } catch {
            status = "Load failed: \(error)"
        }
        busy = false
    }

    func speak(_ text: String) async {
        guard let tts, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        busy = true
        do {
            if streaming {
                status = "Synthesizing (streaming)…"
                beginPlayback()
                let stats = try await tts.synthesizeStreaming(text) { [weak self] chunk in
                    await self?.scheduleChunk(chunk)
                }
                flushPreroll()
                guard stats.samples > 0 else { status = "Nothing to speak."; busy = false; return }
                status = String(format: "🟢 First audio %.2f s · %.2f s speech in %.2f s (RTF %.2f).",
                                stats.firstChunkSeconds, stats.audioSeconds, stats.totalSeconds,
                                stats.realTimeFactor)
            } else {
                status = "Synthesizing (whole clip)…"
                let t0 = Date()
                let audio = try await tts.synthesize(text)
                let gen = Date().timeIntervalSince(t0)
                guard !audio.isEmpty else { status = "Nothing to speak."; busy = false; return }
                let dur = Double(audio.count) / sr
                player.reset(sampleRate: sr)
                player.play(audio, sampleRate: sr)
                status = String(format: "🔴 Silent for %.2f s, then plays · %.2f s speech (RTF %.2f).",
                                gen, dur, dur > 0 ? gen / dur : 0)
            }
        } catch {
            status = "Error: \(error)"
        }
        busy = false
    }

    private func beginPlayback() { player.reset(sampleRate: sr); preroll.removeAll(); prerolling = true }

    private func scheduleChunk(_ chunk: [Float]) {
        if prerolling {
            preroll.append(chunk)
            if preroll.count >= prerollChunks { flushPreroll() }
        } else {
            player.play(chunk, sampleRate: sr)
        }
    }

    private func flushPreroll() {
        guard prerolling else { return }
        for c in preroll { player.play(c, sampleRate: sr) }
        preroll.removeAll(); prerolling = false
    }
}

struct VoxCPM2View: View {
    @StateObject private var vm = VoxCPM2VM()
    @State private var text = "On device speech synthesis, running entirely on your iPhone."

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("VoxCPM2 — 2B, 48 kHz on-device TTS")
                .font(.title2).bold()
            Text("Scaled MiniCPM4 backbone (28L) + LocDiT-12L diffusion + 48 kHz AudioVAE on Core AI. Five bundles + host glue, no network.")
                .font(.callout).foregroundStyle(.secondary)

            if !vm.loaded {
                Picker("Precision", selection: $vm.lm) {
                    Text("int8 (smaller, ~1.6 GB LM)").tag(VoxCPM2Paths.LMPrecision.int8)
                    Text("fp16 (best quality)").tag(VoxCPM2Paths.LMPrecision.fp16)
                }
                .pickerStyle(.segmented)
                .disabled(vm.busy)
                Button { Task { await vm.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(vm.busy)
            } else {
                TextField("Text", text: $text, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
                Toggle(isOn: $vm.streaming) {
                    Text(vm.streaming ? "Streaming (play as it generates)"
                                      : "No streaming (wait for whole clip)")
                        .font(.callout)
                }
                .disabled(vm.busy)
                Button {
                    Task { await vm.speak(text) }
                } label: {
                    Label("Speak", systemImage: "play.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.busy)
            }

            if vm.busy { ProgressView().controlSize(.small) }
            Text(vm.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 420)
        #endif
    }
}
