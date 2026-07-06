// DotsView — the "Voice ML" (multilingual) tab: dots.tts (rednote-hilab, 2B, 24-language) TTS on
// Core AI. Community port. Drives CoreAIKit's `DotsTTS` (four bundles + host glue, all on device),
// the validated Python blueprint conversion/dots_tts/e2e_full.py (engine wav cos 0.9959).
//
// Assets load from a `DotsAssets` root (macOS dev: symlink -> conversion/dots_tts/artifacts, which
// holds dots_*_.aimodel bundle dirs + dots_host_glue/ + tokenizer/). iOS: applicationSupport/DotsAssets.
import CoreAIKit
import SwiftUI

enum DotsAssets {
    static let repo = "mlboydaisuke/dots.tts-CoreAI"   // not yet published

    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("DotsAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DotsAssets")
        #endif
    }

    /// Present only if the tokenizer + glue are staged.
    static var root: URL? {
        let p = location
        let fm = FileManager.default
        return fm.fileExists(atPath: p.appendingPathComponent("tokenizer").path)
            && fm.fileExists(atPath: p.appendingPathComponent("dots_host_glue").path) ? p : nil
    }

    static func paths(root: URL, lm: DotsTTSPaths.LMPrecision = .fp16,
                      decoder: DotsTTSPaths.Decoder = .mf) -> DotsTTSPaths {
        let tok = root.appendingPathComponent("tokenizer")
        #if os(macOS)
        return .standard(artifactsRoot: root, lm: lm, decoder: decoder, tokenizerDir: tok)
        #else
        return .aot(root: root, arch: "h18p", lm: .int4, decoder: decoder, tokenizerDir: tok)  // device: int4 bb, mf decoder (~5.5 GB)
        #endif
    }
}

@MainActor
final class DotsVM: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var streaming = true
    @Published var lm: DotsTTSPaths.LMPrecision = .fp16
    let decoder: DotsTTSPaths.Decoder = .mf   // MeanFlow: ~5× faster than soar

    private var tts: DotsTTS?
    private let player = AudioPlayer()
    private var preroll: [[Float]] = []
    private var prerolling = false
    private let prerollChunks = 3

    private var sr: Double { Double(DotsTTS.sampleRate) }   // 48 kHz

    func load() async {
        busy = true; status = "Loading dots.tts 2B (\(lm.rawValue))…"
        do {
            guard let root = DotsAssets.root else {
                status = "model not found (need tokenizer/ + dots_host_glue/ + dots_* at \(DotsAssets.location.path))."
                busy = false; return
            }
            let t0 = Date()
            tts = try await DotsTTS(paths: DotsAssets.paths(root: root, lm: lm, decoder: decoder), decoder: decoder)
            loaded = true
            status = String(format: "Ready (%@, loaded in %.1f s).", lm.rawValue, Date().timeIntervalSince(t0))
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
        if prerolling { preroll.append(chunk); if preroll.count >= prerollChunks { flushPreroll() } }
        else { player.play(chunk, sampleRate: sr) }
    }
    private func flushPreroll() {
        guard prerolling else { return }
        for c in preroll { player.play(c, sampleRate: sr) }
        preroll.removeAll(); prerolling = false
    }
}

struct DotsView: View {
    @StateObject private var vm = DotsVM()
    @State private var text = "Hello from Core A I."

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("dots.tts — 2B, 24-language on-device TTS")
                .font(.title2).bold()
            Text("Qwen2.5-1.5B backbone + 18L AdaLN flow-matching DiT + 48 kHz BigVGAN on Core AI. Four bundles + host glue, no network. Multilingual.")
                .font(.callout).foregroundStyle(.secondary)

            if !vm.loaded {
                Picker("Precision", selection: $vm.lm) {
                    Text("fp16 (reference quality)").tag(DotsTTSPaths.LMPrecision.fp16)
                    Text("int8 (smaller LM)").tag(DotsTTSPaths.LMPrecision.int8)
                    Text("int4 (smallest)").tag(DotsTTSPaths.LMPrecision.int4)
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
                    Text(vm.streaming ? "Streaming (play as it generates)" : "No streaming (wait for whole clip)")
                        .font(.callout)
                }
                .disabled(vm.busy)
                Button { Task { await vm.speak(text) } } label: {
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
