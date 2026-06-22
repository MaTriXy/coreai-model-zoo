// Headless self-test: KOKORO_SELFTEST=1 runs the Swift TTS pipeline once and writes
// /tmp/kokoro_swift.f32 (raw float32 24 kHz), for the Python parity gate. Reliable
// vs the SwiftUI lifecycle (runs synchronously in App.init).
import Foundation

import CoreAIKit

func runKokoroSelfTest() async {
    NSLog("KOKORO selftest: start")
    do {
        let store = ModelStore()
        var urls: [URL] = []
        for name in KokoroAssets.bundleNames {
            urls.append(try await KokoroAssets.bundle(name, store: store) { _ in })
        }
        let res = KokoroAssets.resources
        let tts = try await KokoroTTS(
            predictor: urls[0], prosody: urls[1], vocoder: urls[2],
            vocab: res.appendingPathComponent("vocab.json"),
            lLinear: res.appendingPathComponent("l_linear.bin"))
        NSLog("KOKORO selftest: bundles loaded")
        let vdir = res.appendingPathComponent("voices")
        for f in (try? FileManager.default.contentsOfDirectory(at: vdir, includingPropertiesForKeys: nil)) ?? []
        where f.pathExtension == "bin" {
            await tts.loadVoice(f.deletingPathExtension().lastPathComponent, url: f)
        }
        let data = try Data(contentsOf: res.appendingPathComponent("demo_phrases.json"))
        let phrases = try JSONDecoder().decode([DemoPhrase].self, from: data)
        let p = phrases[0]
        let t0 = Date()
        let audio = try await tts.synthesize(ids: p.ids, voice: "af_heart")
        let ms = Date().timeIntervalSince(t0) * 1000
        let out = audio.withUnsafeBytes { Data($0) }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try out.write(to: docs.appendingPathComponent("kokoro_swift.f32"))
        NSLog("KOKORO selftest: wrote %d samples (%.0f ms) to %@", audio.count, ms, docs.path)

        // Free-text path: on-device G2P (MisakiSwift) -> phonemes -> audio.
        let freeText = "The weather is lovely today, so let's go for a walk in the park."
        let fids = await tts.ids(forText: freeText)
        let t1 = Date()
        let fa = try await tts.synthesizeText(freeText, voice: "af_bella")
        NSLog("KOKORO selftest: free-text ids=%d audio=%d (%.0f ms)", fids.count, fa.count,
              Date().timeIntervalSince(t1) * 1000)
        try fa.withUnsafeBytes { Data($0) }.write(to: docs.appendingPathComponent("kokoro_freetext.f32"))
    } catch {
        NSLog("KOKORO selftest: FAILED %@", String(describing: error))
    }
}
