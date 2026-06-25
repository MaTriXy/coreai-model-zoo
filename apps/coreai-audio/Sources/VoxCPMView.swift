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
    static func paths(root: URL, lm: VoxCPMPaths.LMPrecision = .int8) -> VoxCPMPaths {
        #if os(macOS)
        return .standard(artifactsRoot: root, lm: lm)
        #else
        return .aot(root: root, arch: "h18p", lm: lm)
        #endif
    }
}

@MainActor
final class VoxCPMVM: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var streaming = true          // off = old behaviour (silent until the whole clip is synthesized)
    @Published var lm: VoxCPMPaths.LMPrecision = .int8   // iPhone: int8 = same RTF as fp16, smaller + fast load + prefill TTFB

    private var tts: VoxCPMTTS?
    private let player = AudioPlayer()
    // Streaming jitter buffer: hold the first few chunks before starting playback so a momentary
    // generation slowdown (RTF near 1.0) can't starve the player → no crackle/underrun.
    private var preroll: [[Float]] = []
    private var prerolling = false
    private let prerollChunks = 2          // ~0.96 s of lead (one chunk ≈ 0.48 s)

    func load() async {
        busy = true; status = "Loading VoxCPM (\(lm == .fp16 ? "fp16" : "int8"))…"
        do {
            guard let root = VoxCPMAssets.root else {
                status = "Model not found at \(VoxCPMAssets.location.path)."
                busy = false; return
            }
            let t0 = Date()
            let t = try await VoxCPMTTS(paths: VoxCPMAssets.paths(root: root, lm: lm))
            tts = t; loaded = true
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
                // NEW: stream chunks as they are decoded; a small pre-roll buffer absorbs RTF jitter.
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
                // OLD: synthesize the whole clip first, then play — silent for the full generation time.
                status = "Synthesizing (whole clip)…"
                let t0 = Date()
                let audio = try await tts.synthesize(text)
                let gen = Date().timeIntervalSince(t0)
                guard !audio.isEmpty else { status = "Nothing to speak."; busy = false; return }
                let dur = Double(audio.count) / Double(VoxCPMTTS.sampleRate)
                player.reset(sampleRate: Double(VoxCPMTTS.sampleRate))
                player.play(audio, sampleRate: Double(VoxCPMTTS.sampleRate))
                status = String(format: "🔴 Silent for %.2f s, then plays · %.2f s speech (RTF %.2f).",
                                gen, dur, dur > 0 ? gen / dur : 0)
            }
        } catch {
            status = "Error: \(error)"
        }
        busy = false
    }

    /// Begin a new streamed utterance: reset the node and arm the pre-roll buffer.
    private func beginPlayback() {
        player.reset(sampleRate: Double(VoxCPMTTS.sampleRate))
        preroll.removeAll(); prerolling = true
    }

    /// Queue one streamed chunk. While pre-rolling, accumulate; once `prerollChunks` are buffered,
    /// flush them and stream the rest directly (gapless). The lead buffer absorbs RTF jitter.
    private func scheduleChunk(_ chunk: [Float]) {
        if prerolling {
            preroll.append(chunk)
            if preroll.count >= prerollChunks { flushPreroll() }
        } else {
            player.play(chunk, sampleRate: Double(VoxCPMTTS.sampleRate))
        }
    }

    /// Flush buffered pre-roll chunks and leave pre-roll mode (also covers clips shorter than the buffer).
    private func flushPreroll() {
        guard prerolling else { return }
        for c in preroll { player.play(c, sampleRate: Double(VoxCPMTTS.sampleRate)) }
        preroll.removeAll(); prerolling = false
    }

    /// One-tap OLD-vs-NEW speed comparison (same int8 model, zero quality difference — speed only):
    ///   OLD = no prefill bundle + non-streaming = the originally shipped behaviour (silent until the
    ///         whole clip is synthesized, so first-audio == full generation time).
    ///   NEW = int8 + q=32 prefill bundle + streaming (audio starts after the first ~0.48 s chunk; NEW is
    ///         also played back so you hear it). Reports first-audio + RTF for both, and the TTFB speedup.
    func compareOldVsNew(_ text: String) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard let root = VoxCPMAssets.root else {
            status = "Model not found at \(VoxCPMAssets.location.path)."; return
        }
        busy = true
        let sr = Double(VoxCPMTTS.sampleRate)
        var oldFirst = 0.0, oldRTF = 0.0, newFirst = 0.0, newRTF = 0.0
        do {
            // OLD: int8, prefill bundle disabled, non-streaming (originally shipped).
            status = "Speed test — OLD (no prefill, non-streaming)…"
            var oldPaths = VoxCPMAssets.paths(root: root, lm: .int8)
            oldPaths.prefillBase = nil; oldPaths.prefillRes = nil
            let oldTTS = try await VoxCPMTTS(paths: oldPaths)
            let t0 = Date()
            let audio = try await oldTTS.synthesize(text)              // first audio == full gen
            oldFirst = Date().timeIntervalSince(t0)
            oldRTF = oldFirst / (Double(audio.count) / sr)
        } catch { status = "OLD failed: \(error)"; busy = false; return }
        do {
            // NEW: int8 + prefill bundle + streaming (and play it back).
            status = "Speed test — NEW (prefill + streaming)…"
            let newTTS = try await VoxCPMTTS(paths: VoxCPMAssets.paths(root: root, lm: .int8))
            beginPlayback()
            let stats = try await newTTS.synthesizeStreaming(text) { [weak self] chunk in
                await self?.scheduleChunk(chunk)
            }
            flushPreroll()
            newFirst = stats.firstChunkSeconds; newRTF = stats.realTimeFactor
        } catch { status = "NEW failed: \(error)"; busy = false; return }
        let ttfbX = newFirst > 0 ? oldFirst / newFirst : 0
        status = String(format: "OLD: first %.2fs · RTF %.2f  →  NEW: first %.2fs · RTF %.2f  (TTFB %.1f× faster)",
                        oldFirst, oldRTF, newFirst, newRTF, ttfbX)
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
                Picker("Precision", selection: $vm.lm) {
                    Text("fp16 (fast, best quality)").tag(VoxCPMPaths.LMPrecision.fp16)
                    Text("int8 (smaller)").tag(VoxCPMPaths.LMPrecision.int8)
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
                                      : "No streaming (old: wait for whole clip)")
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

            Button {
                Task { await vm.compareOldVsNew(text) }
            } label: {
                Label("Speed test: OLD vs NEW", systemImage: "stopwatch").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(vm.busy)

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
