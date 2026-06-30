// MusicGenModel — the "Music" tab VM. Drives CoreAIKit's `StableAudioMusic` (Stable Audio Open Small,
// zoo's first on-device music generation) and plays the result via AudioPlayer.
//
// Assets: `StableAudioAssets` root (dev symlink -> conversion ship_macos) holds the 3 .aimodel +
// metadata.json + t5_tokenizer/. macOS uses `.aimodel`; device uses the AOT `.h18p.aimodelc`.
import CoreAIKit
import Foundation
import SwiftUI

enum StableAudioAssets {
    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("StableAudioAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("StableAudioAssets")
        #endif
    }
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("metadata.json").path) ? p : nil
    }
}

@MainActor
final class MusicGenModel: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var lastStats = ""

    private var engine: StableAudioMusic?
    private let player = AudioPlayer()
    private var lastAudio: [Float] = []                 // [2 * N], ch0 then ch1
    private var sr: Double { Double(StableAudioMusic.sampleRate) }
    private var N: Int { StableAudioMusic.audioSamples }

    func load() async {
        busy = true; defer { busy = false }
        guard let root = StableAudioAssets.root else {
            status = "Model not found — stage the bundles at \(StableAudioAssets.location.path)"; return
        }
        status = "Loading Stable Audio (3 bundles)…"
        do {
            #if os(macOS)
            let paths = StableAudioPaths.resolve(root: root, aot: false)
            #else
            let paths = StableAudioPaths.resolve(root: root, aot: true)
            #endif
            engine = try await StableAudioMusic(paths: paths, computeUnits: .gpu)
            loaded = true
            status = "Ready. Describe music (e.g. \"128 BPM tech house drum loop\")."
        } catch {
            status = "Load failed: \(error)"
        }
    }

    func generate(prompt: String, seconds: Float) async {
        guard let engine, !busy else { return }
        busy = true; defer { busy = false }
        status = "Generating…"
        do {
            let clock = ContinuousClock(); let t0 = clock.now
            let audio = try await engine.generate(prompt: prompt, seconds: seconds)
            let el = clock.now - t0
            let ms = Double(el.components.seconds) * 1000 + Double(el.components.attoseconds) / 1e15
            lastAudio = audio

            var mono = [Float](repeating: 0, count: N)
            for n in 0..<N { mono[n] = 0.5 * (audio[n] + audio[N + n]) }   // downmix for playback
            player.reset(sampleRate: sr)
            player.play(mono, sampleRate: sr)

            lastStats = String(format: "%.1f s in %.2f s (%.0f× real-time)", Double(N) / sr, ms / 1000, (Double(N) / sr) / (ms / 1000))
            status = "Done — \(lastStats)"
        } catch {
            status = "Generation failed: \(error)"
        }
    }

    func stop() { player.reset(sampleRate: sr) }

    func saveWav(to url: URL) {
        guard lastAudio.count == 2 * N else { return }
        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
        let ch = 2, bytes = 2, dataLen = N * ch * bytes, srI = StableAudioMusic.sampleRate
        d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(UInt16(ch)); u32(UInt32(srI))
        u32(UInt32(srI * ch * bytes)); u16(UInt16(ch * bytes)); u16(16)
        d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
        for n in 0..<N { for c in 0..<ch { let s = max(-1, min(1, lastAudio[c * N + n])); u16(UInt16(bitPattern: Int16(s * 32767))) } }
        try? d.write(to: url)
    }
}
