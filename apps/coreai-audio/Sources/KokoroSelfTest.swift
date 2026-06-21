// Headless self-test: KOKORO_SELFTEST=1 runs the Swift TTS pipeline once and writes
// /tmp/kokoro_swift.f32 (raw float32 24 kHz), for the Python parity gate. Reliable
// vs the SwiftUI lifecycle (runs synchronously in App.init).
import Foundation

func runKokoroSelfTest() async {
    NSLog("KOKORO selftest: start")
    let assets = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("KokoroAssets")
    do {
        let tts = try await KokoroTTS(assets: assets)
        NSLog("KOKORO selftest: bundles loaded")
        let vdir = assets.appendingPathComponent("voices")
        for f in (try? FileManager.default.contentsOfDirectory(at: vdir, includingPropertiesForKeys: nil)) ?? []
        where f.pathExtension == "bin" {
            await tts.loadVoice(f.deletingPathExtension().lastPathComponent, url: f)
        }
        let data = try Data(contentsOf: assets.appendingPathComponent("demo_phrases.json"))
        let phrases = try JSONDecoder().decode([DemoPhrase].self, from: data)
        let p = phrases[0]
        let t0 = Date()
        let audio = try await tts.synthesize(ids: p.ids, voice: "af_heart")
        let ms = Date().timeIntervalSince(t0) * 1000
        let out = audio.withUnsafeBytes { Data($0) }
        try out.write(to: URL(fileURLWithPath: "/tmp/kokoro_swift.f32"))
        NSLog("KOKORO selftest: wrote %d samples (%.0f ms)", audio.count, ms)
    } catch {
        NSLog("KOKORO selftest: FAILED %@", String(describing: error))
    }
}
