// Headless self-test: VOXCPM_SELFTEST=1 loads VoxCPM once, synthesizes a phrase, and writes
// /Documents/voxcpm_swift.f32 (raw float32, 16 kHz mono) + logs the real-time factor. Lets the
// device build be verified without the UI (cf. KokoroSelfTest / ASRSelfTest).
import CoreAIKit
import Foundation

func runVoxCPMSelfTest() async {
    NSLog("VOXCPM selftest: start")
    do {
        guard let root = VoxCPMAssets.root else {
            NSLog("VOXCPM selftest: no model at %@", VoxCPMAssets.location.path)
            return
        }
        let t0 = Date()
        let tts = try await VoxCPMTTS(paths: VoxCPMAssets.paths(root: root))
        NSLog("VOXCPM selftest: loaded 5 bundles + glue (%.1f s)", Date().timeIntervalSince(t0))

        let text = "On device speech synthesis, running entirely on your iPhone."
        let t1 = Date()
        let audio = try await tts.synthesize(text)
        let elapsed = Date().timeIntervalSince(t1)
        let dur = Double(audio.count) / Double(VoxCPMTTS.sampleRate)
        NSLog("VOXCPM selftest: %d samples = %.2f s audio in %.2f s (RTF %.2f)",
              audio.count, dur, elapsed, dur > 0 ? elapsed / dur : 0)

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try audio.withUnsafeBytes { Data($0) }.write(to: docs.appendingPathComponent("voxcpm_swift.f32"))
        NSLog("VOXCPM selftest: wrote voxcpm_swift.f32 to %@", docs.path)
    } catch {
        NSLog("VOXCPM selftest: FAILED %@", String(describing: error))
    }
}
