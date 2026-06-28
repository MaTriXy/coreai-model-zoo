// Headless VoxCPM2 (2B) sanity run: VOXCPM2_SELFTEST=1. Loads the int8 model, synthesizes one
// sentence (streaming), logs first-audio + RTF, and writes a 48 kHz WAV to Documents so the whole
// on-device Swift pipeline (5 bundles + host glue) can be verified end-to-end without the GUI.
import CoreAIKit
import Foundation

private final class ChunkSink2: @unchecked Sendable {
    var audio: [Float] = []
    func add(_ chunk: [Float]) { audio.append(contentsOf: chunk) }
}

private func writeWav48k(_ samples: [Float], to url: URL) throws {
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

func runVoxCPM2SelfTest() async {
    NSLog("VOXCPM2 selftest: start")
    guard let root = VoxCPM2Assets.root else {
        NSLog("VOXCPM2 selftest: no v2 model at %@ (need tokenizer2/ + voxcpm2_*)", VoxCPM2Assets.location.path)
        return
    }
    let text = "On device speech synthesis, running entirely on your iPhone."
    do {
        let t0 = Date()
        let tts = try await VoxCPM2TTS(paths: VoxCPM2Assets.paths(root: root, lm: .int8))
        NSLog("VOXCPM2: loaded int8 in %.1f s", Date().timeIntervalSince(t0))
        let sink = ChunkSink2()
        let stats = try await tts.synthesizeStreaming(text) { sink.add($0) }
        NSLog("VOXCPM2: first audio %.2f s | %.2f s speech in %.2f s | RTF %.2f | %d samples",
              stats.firstChunkSeconds, stats.audioSeconds, stats.totalSeconds, stats.realTimeFactor, stats.samples)
        let a = sink.audio
        if !a.isEmpty {
            let rms = (a.reduce(0) { $0 + $1 * $1 } / Float(a.count)).squareRoot()
            let peak = a.map { abs($0) }.max() ?? 0
            NSLog("VOXCPM2: rms %.3f peak %.3f", rms, peak)
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let url = docs.appendingPathComponent("voxcpm2_app.wav")
            try writeWav48k(a, to: url)
            NSLog("VOXCPM2: wrote %@", url.path)
        }
    } catch {
        NSLog("VOXCPM2 selftest: FAILED %@", String(describing: error))
    }
    NSLog("VOXCPM2 selftest: done")
}
