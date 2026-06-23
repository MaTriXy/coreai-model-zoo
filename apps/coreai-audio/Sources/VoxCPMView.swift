// VoxCPMView — the "Voice" tab: VoxCPM-0.5B text-to-speech on Core AI (the zoo's voice-capable TTS:
// MiniCPM4 TSLM/RALM backbone + LocDiT diffusion + AudioVAE, all on the engine; tiny projections
// host-side). Drives CoreAIKit's `VoxCPMTTS`. Plain TTS (fixed speaker); voice-clone is a follow-on.
//
// Model assets (5 bundles + host glue + tokenizer) load from a local `VoxCPMAssets` root in dev
// (symlink the conversion `artifacts/` dir there), else download from HF once the repo is published.
import CoreAIKit
import SwiftUI

enum VoxCPMAssets {
    static let repo = "mlboydaisuke/VoxCPM-0.5B-CoreAI"

    /// The model root for this platform, or nil if not present.
    /// - macOS (dev): a `VoxCPMAssets/` symlink next to this file → conversion `artifacts/`
    ///   (the JIT `.aimodel` bundles).
    /// - iOS: `Library/Application Support/VoxCPMAssets`, sideloaded with the AOT `.aimodelc`
    ///   bundles (+ `voxcpm_host_glue/` + `tokenizer/`) via `devicectl ... copy`.
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("tokenizer").path) ? p : nil
    }

    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("VoxCPMAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VoxCPMAssets")
        #endif
    }

    /// Platform-correct bundle paths (macOS JIT `.aimodel` vs iOS AOT `.aimodelc`).
    static func paths(root: URL) -> VoxCPMPaths {
        #if os(macOS)
        return .standard(artifactsRoot: root)
        #else
        return .aot(root: root, arch: "h18p")
        #endif
    }
}

@MainActor
final class VoxCPMVM: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false

    private var tts: VoxCPMTTS?
    private let player = AudioPlayer()

    func load() async {
        busy = true; status = "Loading VoxCPM (5 bundles + glue)…"
        do {
            guard let root = VoxCPMAssets.root else {
                status = "Model not found at \(VoxCPMAssets.location.path)."
                busy = false; return
            }
            let t0 = Date()
            let t = try await VoxCPMTTS(paths: VoxCPMAssets.paths(root: root))
            tts = t; loaded = true
            status = String(format: "Ready (loaded in %.1f s).", Date().timeIntervalSince(t0))
        } catch {
            status = "Load failed: \(error)"
        }
        busy = false
    }

    func speak(_ text: String) async {
        guard let tts, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        busy = true; status = "Synthesizing…"
        do {
            let t0 = Date()
            let audio = try await tts.synthesize(text)
            guard !audio.isEmpty else { status = "Nothing to speak."; busy = false; return }
            let dur = Double(audio.count) / Double(VoxCPMTTS.sampleRate)
            let elapsed = Date().timeIntervalSince(t0)
            status = String(format: "%.2f s audio in %.2f s (RTF %.2f).", dur, elapsed, elapsed / dur)
            player.play(audio, sampleRate: Double(VoxCPMTTS.sampleRate))
        } catch {
            status = "Error: \(error)"
        }
        busy = false
    }
}

struct VoxCPMView: View {
    @StateObject private var vm = VoxCPMVM()
    @State private var text = "On device speech synthesis, running entirely on your iPhone."

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("VoxCPM-0.5B — on-device voice TTS")
                .font(.title2).bold()
            Text("MiniCPM4 backbone + LocDiT diffusion + AudioVAE on Core AI. Five bundles + host glue, no network.")
                .font(.callout).foregroundStyle(.secondary)

            if !vm.loaded {
                Button { Task { await vm.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(vm.busy)
            } else {
                TextField("Text", text: $text, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
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
