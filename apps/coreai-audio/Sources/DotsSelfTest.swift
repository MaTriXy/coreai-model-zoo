// Headless dots.tts sanity run: DOTS_SELFTEST=1. Loads the fp16 model, synthesizes one sentence,
// logs RTF + rms/peak, writes a 48 kHz WAV to Documents — verifies the whole on-device Swift pipeline
// (4 bundles + host glue) end-to-end without the GUI. The Swift solver noise differs from the torch
// oracle (own RNG), so this checks for VALID speech (right length, non-silent), not a bit-match.
import CoreAIKit
import Foundation

private final class DotsChunkSink: @unchecked Sendable {
    var audio: [Float] = []
    func add(_ chunk: [Float]) { audio.append(contentsOf: chunk) }
}

private func dotsWriteWav48k(_ samples: [Float], to url: URL) throws {
    let sr: UInt32 = 48_000
    var pcm = [Int16](repeating: 0, count: samples.count)
    for i in samples.indices { pcm[i] = Int16(max(-1, min(1, samples[i])) * 32767) }
    var d = Data()
    func u32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    func u16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    let bytes = UInt32(pcm.count * 2)
    d.append("RIFF".data(using: .ascii)!); d.append(u32(36 + bytes)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); d.append(u32(16)); d.append(u16(1)); d.append(u16(1))
    d.append(u32(sr)); d.append(u32(sr * 2)); d.append(u16(2)); d.append(u16(16))
    d.append("data".data(using: .ascii)!); d.append(u32(bytes))
    pcm.withUnsafeBytes { d.append(contentsOf: $0) }
    try d.write(to: url)
}

func runDotsSelfTest() async {
    NSLog("DOTS selftest: start")
    guard let root = DotsAssets.root else {
        NSLog("DOTS selftest: no model at %@ (need tokenizer/ + dots_host_glue/ + dots_*)", DotsAssets.location.path)
        return
    }
    let text = "Hello from Core A I."
    do {
        let t0 = Date()
        let tts = try await DotsTTS(paths: DotsAssets.paths(root: root, lm: .fp16, decoder: .mf), decoder: .mf)
        NSLog("DOTS: loaded (mf) in %.1f s", Date().timeIntervalSince(t0))
        let sink = DotsChunkSink()
        let stats = try await tts.synthesizeStreaming(text) { sink.add($0) }
        NSLog("DOTS: first audio %.2f s | %.2f s speech in %.2f s | RTF %.2f | %d samples",
              stats.firstChunkSeconds, stats.audioSeconds, stats.totalSeconds, stats.realTimeFactor, stats.samples)
        let prof = tts.profile.sorted { $0.value > $1.value }
            .map { String(format: "%@=%.2fs", $0.key, $0.value) }.joined(separator: " ")
        NSLog("DOTS profile: %@", prof)
        let a = sink.audio
        if !a.isEmpty {
            let rms = (a.reduce(0) { $0 + $1 * $1 } / Float(a.count)).squareRoot()
            let peak = a.map { abs($0) }.max() ?? 0
            NSLog("DOTS: rms %.3f peak %.3f", rms, peak)
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let url = docs.appendingPathComponent("dots_app.wav")
            try dotsWriteWav48k(a, to: url)
            NSLog("DOTS: wrote %@", url.path)
        }
    } catch {
        NSLog("DOTS selftest: FAILED %@", String(describing: error))
    }
    NSLog("DOTS selftest: done")
}
