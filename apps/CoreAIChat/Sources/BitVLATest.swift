// On-device BitVLA self-test (BITVLA_TEST=1): load the vision + LLM .aimodelc + device data,
// run the bundled COCO sample through preset 0 ("pick up the remote" = the oracle prompt), and
// print the 7-DoF action + per-stage timing. Confirms the 1.58-bit ternary VLA runs end-to-end on
// the iPhone GPU. Preset 0 should land near the Mac/official oracle action.

import CoreGraphics
import Foundation
import ImageIO

@MainActor
enum BitVLATest {
    static func run() async {
        print("[bitvla] ====== on-device BitVLA (1.58-bit VLA) self-test ======")
        let backend = BitVLABackend()
        do {
            try await backend.load()
            print("[bitvla] loaded. presets=\(backend.presetTexts)")

            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let imgURL = docs.appendingPathComponent("models")
                .appendingPathComponent(BitVLABackend.dataDir).appendingPathComponent("sample.png")
            guard let src = CGImageSourceCreateWithURL(imgURL as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                print("[bitvla] MISSING sample.png at \(imgURL.path)"); return
            }
            print("[bitvla] sample image \(cg.width)x\(cg.height); predicting preset 0 ...")
            let r = try await backend.predict(cgImage: cg, presetIndex: 0)
            print("[bitvla] action tokens: \(r.tokens)")
            print("[bitvla] 7-DoF (bridge_orig): \(r.dof.map { String(format: "%.4f", $0) })")
            print(String(format: "[bitvla] timing: vision %.0fms | prefill %.0fms | decode %.0fms",
                         r.visionMs, r.prefillMs, r.decodeMs))
            print("[bitvla] oracle ids (official): [128012, 128131, 128012, 128012, 128267, 128267, 128012]")
            print("[bitvla] DONE — BitVLA runs on iPhone GPU.")
        } catch {
            print("[bitvla] ERROR: \(error)")
        }
    }
}
