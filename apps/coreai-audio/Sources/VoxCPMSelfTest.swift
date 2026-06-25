// Headless on-device speed comparison: VOXCPM_SELFTEST=1.
//   OLD = int8, no prefill bundle, NON-streaming (= the originally shipped behaviour: silent until the
//         whole clip is synthesized, so time-to-first-audio == full generation time).
//   NEW = int8 + q=32 prefill bundle + streaming (audio starts after the first ~0.48s chunk).
// Same model/precision/RTF — the comparison isolates the perceived-latency win. Logs both + the ratio.
import CoreAIKit
import Foundation

private final class ChunkSink: @unchecked Sendable {
    var audio: [Float] = []
    func add(_ chunk: [Float]) { audio.append(contentsOf: chunk) }
}

func runVoxCPMSelfTest() async {
    NSLog("VOXCPM selftest: start (OLD vs NEW speed comparison)")
    guard let root = VoxCPMAssets.root else {
        NSLog("VOXCPM selftest: no model at %@", VoxCPMAssets.location.path)
        return
    }
    let text = "On device speech synthesis, running entirely on your iPhone."
    let sr = Double(VoxCPMTTS.sampleRate)

    var oldFirst = 0.0, oldRTF = 0.0, newFirst = 0.0, newRTF = 0.0
    do {
        // OLD: int8, prefill bundle disabled, non-streaming (synthesize the whole clip, then it's ready).
        var oldPaths = VoxCPMAssets.paths(root: root, lm: .int8)
        oldPaths.prefillBase = nil; oldPaths.prefillRes = nil
        let oldTTS = try await VoxCPMTTS(paths: oldPaths)
        let t0 = Date()
        let audio = try await oldTTS.synthesize(text)
        oldFirst = Date().timeIntervalSince(t0)                 // non-streaming: first audio == full gen
        oldRTF = oldFirst / (Double(audio.count) / sr)
        NSLog("VOXCPM OLD (int8, no-prefill, non-streaming): first audio %.2f s (= full gen) | RTF %.2f", oldFirst, oldRTF)
    } catch { NSLog("VOXCPM OLD: FAILED %@", String(describing: error)) }

    do {
        // NEW: int8 + prefill bundle + streaming.
        let newTTS = try await VoxCPMTTS(paths: VoxCPMAssets.paths(root: root, lm: .int8))
        let sink = ChunkSink()
        let stats = try await newTTS.synthesizeStreaming(text) { sink.add($0) }
        newFirst = stats.firstChunkSeconds; newRTF = stats.realTimeFactor
        NSLog("VOXCPM NEW (int8 + prefill + streaming): first audio %.2f s | %.2f s speech in %.2f s | RTF %.2f",
              newFirst, stats.audioSeconds, stats.totalSeconds, newRTF)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try sink.audio.withUnsafeBytes { Data($0) }.write(to: docs.appendingPathComponent("voxcpm_new.f32"))
    } catch { NSLog("VOXCPM NEW: FAILED %@", String(describing: error)) }

    if oldFirst > 0, newFirst > 0 {
        NSLog("VOXCPM ===> first-audio %.1fx faster (%.2f s -> %.2f s); RTF %.2f -> %.2f (same model)",
              oldFirst / newFirst, oldFirst, newFirst, oldRTF, newRTF)
    }
    NSLog("VOXCPM selftest: done")
}
