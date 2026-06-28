// Headless self-test: PARAKEET_SELFTEST=1 assembles a clean Parakeet bundle from the local
// conversion artifacts, transcribes the libri1 wav at PARAKEET_SELFTEST_WAV, and writes the result
// to /tmp/parakeet_swift_result.txt + NSLog. Also dumps the Swift mel ([128,2885] f32) to
// /tmp/parakeet_swift_mel.bin so it can be diffed against oracle_30s input_features. Verifies the
// CoreAIKit Parakeet path (KitParakeetModel) end-to-end on Mac without driving the GUI.

import CoreAIKit
import Foundation

private let kGold = "With her white paint and her scarlet smokestack, the Inverashiel"

func runParakeetSelfTest() async {
    NSLog("Parakeet selftest: start")
    let env = ProcessInfo.processInfo.environment
    let wav = env["PARAKEET_SELFTEST_WAV"] ?? "/tmp/libri1.wav"
    let artifacts = URL(filePath:
        "/Users/majimadaisuke/code/coreai/coreai-models-community/conversion/parakeet/artifacts")
    let out = "/tmp/parakeet_swift_result.txt"

    func write(_ s: String) {
        try? s.write(toFile: out, atomically: true, encoding: .utf8)
        NSLog("Parakeet selftest: %@", s)
    }

    do {
        // Assemble a clean bundle: the dev artifacts dir holds two encoders (L1485 + L2885); the
        // shipped bundle has exactly one, so stage symlinks to the 30 s encoder + predict + joint
        // and copy the tokenizer next to them.
        let fm = FileManager.default
        let bundle = URL(filePath: NSTemporaryDirectory()).appending(path: "parakeet_bundle")
        try? fm.removeItem(at: bundle)
        try fm.createDirectory(at: bundle, withIntermediateDirectories: true)
        func link(_ src: String, _ dst: String) throws {
            try fm.createSymbolicLink(
                at: bundle.appending(path: dst), withDestinationURL: artifacts.appending(path: src))
        }
        try link("parakeet_encoder_float16_L2885.aimodel", "encoder.aimodel")
        try link("parakeet_predict_float32.aimodel", "predict.aimodel")
        try link("parakeet_joint_float32.aimodel", "joint.aimodel")
        for f in ["tokenizer.json", "tokenizer_config.json"] {
            try? fm.copyItem(
                at: artifacts.appending(path: "bundle_assets/\(f)"), to: bundle.appending(path: f))
        }

        let t0 = Date()
        let model = try await KitParakeetModel(bundleAt: bundle)
        NSLog("Parakeet selftest: model loaded (%.1fs)", Date().timeIntervalSince(t0))

        guard let pcm = AudioLoader.load16kMono(URL(filePath: wav)) else {
            write("FAIL: could not decode \(wav)")
            return
        }
        NSLog("Parakeet selftest: clip %@ = %.2fs", wav, Double(pcm.count) / 16000)

        // Dump the Swift mel for a numerical diff vs the golden oracle (mel discipline).
        let melProc = try ParakeetMelPreprocessor.parakeet()
        let melData = melProc.logMel(pcm).withUnsafeBufferPointer { Data(buffer: $0) }
        try? melData.write(to: URL(filePath: "/tmp/parakeet_swift_mel.bin"))
        NSLog("Parakeet selftest: dumped mel [128,%d] -> /tmp/parakeet_swift_mel.bin", melProc.bucketFrames)

        let t1 = Date()
        let result = try await model.transcribe(samples: pcm)
        let ms = Date().timeIntervalSince(t1) * 1000
        let pass = result.text.contains(kGold)
        write("\(pass ? "PASS" : "MISMATCH")  (\(Int(ms)) ms)\n\(result.text)")
    } catch {
        write("FAIL: \(error)")
    }
}
