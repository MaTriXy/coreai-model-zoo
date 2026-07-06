// SeparateModel — the "Separate" tab VM. Drives MelBandSeparator (Mel-Band RoFormer,
// zoo's first source separation) to split a song into vocals + instrumental, and plays
// either stem. Pairs with the Music tab (generate a track, then rip its stems).
//
// Assets: `SeparateAssets` root (dev symlink -> conversion/melband_roformer/ship_macos)
// holds mbr_full_fp16.aimodel (macOS) / .h18p.aimodelc (device) + metadata + golden_*.f32.
import CoreAIKitVision
import Foundation
import SwiftUI

enum SeparateAssets {
    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("SeparateAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SeparateAssets")
        #endif
    }
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("metadata.json").path) ? p : nil
    }
    static var modelURL: URL? {
        guard let r = root else { return nil }
        #if os(macOS)
        return r.appendingPathComponent("mbr_full_fp16.aimodel")
        #else
        return r.appendingPathComponent("mbr_full_fp16.h18p.aimodelc")
        #endif
    }
    /// 8 s stereo demo clip (golden_raw.f32 = ch0 then ch1, C samples each).
    static func demoStereo() -> [[Float]]? {
        guard let r = root,
              let d = try? Data(contentsOf: r.appendingPathComponent("golden_raw.f32")) else { return nil }
        let all = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let C = MelBandSeparator.C
        guard all.count >= 2 * C else { return nil }
        return [Array(all[0..<C]), Array(all[C..<2 * C])]
    }
}

@MainActor
final class SeparateModel: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var progress = 0.0
    @Published var stats = ""
    @Published var haveResult = false

    private var sep: MelBandSeparator?
    private let player = AudioPlayer()
    private var vocals: [[Float]] = []
    private var instrumental: [[Float]] = []
    private let sr = Double(MelBandSeparator.sr)

    func load() async {
        busy = true; defer { busy = false }
        guard let url = SeparateAssets.modelURL else {
            status = "Model not found — stage the bundle at \(SeparateAssets.location.path)"; return
        }
        status = "Loading Mel-Band RoFormer…"
        do {
            sep = try await MelBandSeparator(model: url, computeUnits: .gpu)
            loaded = true
            status = "Ready. Choose a song or use the demo clip."
        } catch { status = "Load failed: \(error)" }
    }

    /// Separate `mix` ([2][T]); if nil, use the bundled 8 s demo.
    func separate(_ mix: [[Float]]?) async {
        guard let sep, !busy else { return }
        guard let input = mix ?? SeparateAssets.demoStereo() else { status = "No audio."; return }
        busy = true; haveResult = false; progress = 0; defer { busy = false }
        status = "Separating…"
        do {
            let clock = ContinuousClock(); let t0 = clock.now
            let (v, i) = try await sep.separate(input) { p in Task { @MainActor in self.progress = p } }
            let el = clock.now - t0
            let ms = Double(el.components.seconds) * 1000 + Double(el.components.attoseconds) / 1e15
            vocals = v; instrumental = i; haveResult = true
            let dur = Double(input[0].count) / sr
            stats = String(format: "%.0f s in %.1f s (%.1f× real-time)", dur, ms / 1000, dur / (ms / 1000))
            status = "Done — \(stats). Play a stem."
            play(vocals)
        } catch { status = "Separation failed: \(error)" }
    }

    func playVocals() { if haveResult { play(vocals) } }
    func playInstrumental() { if haveResult { play(instrumental) } }
    func stop() { player.reset(sampleRate: sr) }

    private func play(_ stereo: [[Float]]) {
        let n = stereo[0].count
        var mono = [Float](repeating: 0, count: n)
        for k in 0..<n { mono[k] = 0.5 * (stereo[0][k] + stereo[1][k]) }
        player.reset(sampleRate: sr)
        player.play(mono, sampleRate: sr)
    }

    func saveWav(_ which: Bool, to url: URL) {   // true = vocals, false = instrumental
        let s = which ? vocals : instrumental
        guard s.count == 2, s[0].count > 0 else { return }
        let n = s[0].count, ch = 2, bytes = 2, dataLen = n * ch * bytes, srI = MelBandSeparator.sr
        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(UInt16(ch)); u32(UInt32(srI))
        u32(UInt32(srI * ch * bytes)); u16(UInt16(ch * bytes)); u16(16)
        d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
        for k in 0..<n { for c in 0..<ch { let x = max(-1, min(1, s[c][k])); u16(UInt16(bitPattern: Int16(x * 32767))) } }
        try? d.write(to: url)
    }
}
