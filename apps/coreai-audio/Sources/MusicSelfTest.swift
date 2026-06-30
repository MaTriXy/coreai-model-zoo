// Headless self-test for the Music tab (MUSIC_SELFTEST=1): load Stable Audio, generate one clip,
// write a wav + print stats. Runs without the GUI (init()-launched).
import CoreAIKit
import Foundation

func runMusicSelfTest() async {
    func log(_ s: String) { print("[MUSIC] \(s)") }
    guard let root = StableAudioAssets.root else {
        log("FAIL: assets not found at \(StableAudioAssets.location.path)"); return
    }
    do {
        let t0 = ContinuousClock().now
        #if os(macOS)
        let paths = StableAudioPaths.resolve(root: root, aot: false)
        #else
        let paths = StableAudioPaths.resolve(root: root, aot: true)   // device uses the AOT .aimodelc
        #endif
        let engine = try await StableAudioMusic(paths: paths, computeUnits: .gpu)
        log("loaded (3 bundles + T5 tokenizer) in \(ms(since: t0))")

        let prompt = ProcessInfo.processInfo.environment["MUSIC_PROMPT"] ?? "128 BPM tech house drum loop"
        let g0 = ContinuousClock().now
        let audio = try await engine.generate(prompt: prompt, seconds: 11)
        let gms = msVal(since: g0)
        let N = StableAudioMusic.audioSamples
        var peak: Float = 0; for v in audio { peak = max(peak, abs(v)) }
        log(String(format: "generated \"%@\": %.1fs audio in %.2fs (%.0f× real-time), peak=%.3f",
                   prompt, Double(N) / Double(StableAudioMusic.sampleRate), gms / 1000,
                   (Double(N) / Double(StableAudioMusic.sampleRate)) / (gms / 1000), peak))

        // write wav
        let out = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MUSIC_OUT"] ?? "/tmp/music_selftest.wav")
        writeWav(audio, n: N, sr: StableAudioMusic.sampleRate, to: out)
        log("wrote \(out.path)")
        log("PASS")
    } catch { log("FAIL: \(error)") }
}

private func ms(since t: ContinuousClock.Instant) -> String { String(format: "%.2fs", msVal(since: t) / 1000) }
private func msVal(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock().now - t
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

private func writeWav(_ a: [Float], n: Int, sr: Int, to url: URL) {
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    let ch = 2, bytes = 2, dataLen = n * ch * bytes
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(UInt16(ch)); u32(UInt32(sr))
    u32(UInt32(sr * ch * bytes)); u16(UInt16(ch * bytes)); u16(16)
    d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
    for i in 0..<n { for c in 0..<ch { let s = max(-1, min(1, a[c * n + i])); u16(UInt16(bitPattern: Int16(s * 32767))) } }
    try? d.write(to: url)
}
