// Headless self-test for the Diarize path (DIARIZE_SELFTEST=1) — the Swift mirror of
// conversion/sortformer_diar/gate_e2e_engine.py + gate_long.py. Two gates on two clips (21.5 s demo,
// 64.5 s = demo×3 which exercises AOSC compress ~4×):
//   1. MEL gate:  Swift NeMo-mel(wav) vs the captured golden mel (cos).
//   2. LOOP gate: drive the SHIPPED fp16 .aimodel (Mac GPU) over the golden mel through the full
//                 streaming host loop, compare per-frame activity to NeMo forward_streaming
//                 (activity-agree @ 0.5 — the actual diarization decision). PASS = ≥99%.
// Plus an end-to-end line (Swift-mel -> loop) for information. Runs GUI-less (init()-launched).
import Foundation

func runDiarizeSelfTest() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let logURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DIAR_RESULT"] ?? "/tmp/diar_result.txt")
    try? "".write(to: logURL, atomically: true, encoding: .utf8)
    func log(_ s: String) {
        print("[DIAR] \(s)")
        if let h = try? FileHandle(forWritingTo: logURL) {
            h.seekToEndOfFile(); h.write(Data(("[DIAR] \(s)\n").utf8)); try? h.close()
        }
    }
    func finish(_ code: Int32) -> Never { log("EXIT \(code)"); exit(code) }

    guard let root = DiarizeAssets.root, let murl = DiarizeAssets.modelURL,
          let filters = DiarizeAssets.melFilters() else {
        log("FAIL: assets not found at \(DiarizeAssets.location.path)"); finish(2)
    }
    guard let wav = AudioLoader.load16kMono(root.appendingPathComponent("test_multispk_16k.wav")) else {
        log("FAIL: demo wav missing"); finish(2)
    }
    log("wav \(wav.count) samples (\(String(format: "%.1f", Double(wav.count) / 16000))s)")

    do {
        let t0 = ContinuousClock().now
        let diar = try await SortformerDiarizer(model: murl, melFilters: filters, computeUnits: .gpu)
        log(String(format: "loaded model in %.2fs", secs(since: t0)))

        var allPass = true
        // (clip label, samples, golden-mel file, golden-preds file)
        let clips: [(String, [Float], String, String)] = [
            ("demo 21.5s", wav, "golden_mel_128xT.f32", "golden_total_preds.f32"),
            ("long 64.5s", wav + wav + wav, "golden_long_mel_128xT.f32", "golden_long_total_preds.f32"),
        ]
        for (label, samples, melFile, predFile) in clips {
            guard let goldMel = DiarizeAssets.f32(melFile), let goldPreds = DiarizeAssets.f32(predFile) else {
                log("FAIL[\(label)]: golden files missing"); allPass = false; continue
            }
            let T = goldMel.count / 128
            let nOutGold = goldPreds.count / 4

            // 1. MEL gate — Swift mel vs golden mel (cos over the aligned [128, minT]).
            let mel = await diar.melForSelfTest(samples)
            let Ts = mel.count / 128
            let cosMel = melCos(mel, Ts, goldMel, T)
            log(String(format: "[\(label)] mel: Swift[128,%d] vs golden[128,%d]  cos %.6f", Ts, T, cosMel))

            // 2. LOOP gate — golden mel -> shipped graph -> host loop, activity-agree vs NeMo.
            let g0 = ContinuousClock().now
            let framesGolden = try await diar.framePreds(mel: goldMel, melFrames: T)
            let dt = secs(since: g0)
            let (agreeG, cosG) = agree(framesGolden, goldPreds, nOutGold)
            let loopPass = agreeG >= 0.99 && framesGolden.count == nOutGold
            log(String(format: "[\(label)] loop(golden mel): frames %d vs %d  cos %.6f  activity-agree %.2f%%  (%.2fs, %.1f× RT)  -> %@",
                       framesGolden.count, nOutGold, cosG, agreeG * 100, dt,
                       Double(samples.count) / 16000 / dt, loopPass ? "PASS" : "FAIL"))

            // end-to-end (Swift mel -> loop): informational — Swift mel ≈ golden, not bit-exact.
            let framesE2E = try await diar.framePreds(mel: mel, melFrames: Ts)
            let (agreeE, _) = agree(framesE2E, goldPreds, min(framesE2E.count, nOutGold))
            let segs = SortformerDiarizer.segments(from: framesE2E)
            log(String(format: "[\(label)] e2e(Swift mel): frames %d  activity-agree %.2f%%  -> %d speaker turns",
                       framesE2E.count, agreeE * 100, segs.count))
            for seg in segs.prefix(6) {
                log(String(format: "      spk%d  %.2f–%.2fs", seg.speaker, seg.startSec, seg.endSec))
            }
            allPass = allPass && loopPass
        }
        log(allPass ? "PASS" : "CHECK (a loop gate is below 99% activity-agree)")
        finish(allPass ? 0 : 3)
    } catch { log("FAIL: \(error)"); finish(4) }
}

/// cos over the aligned [128, minT] region of two mel-major buffers.
private func melCos(_ a: [Float], _ ta: Int, _ b: [Float], _ tb: Int) -> Double {
    let t = min(ta, tb)
    var dot = 0.0, na = 0.0, nb = 0.0
    for m in 0..<128 {
        for i in 0..<t {
            let x = Double(a[m * ta + i]), y = Double(b[m * tb + i])
            dot += x * y; na += x * x; nb += y * y
        }
    }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
}

/// activity-agree @ 0.5 + cos, comparing `frames[n][4]` to golden `[n*4]` frame-major over `n` frames.
private func agree(_ frames: [[Float]], _ gold: [Float], _ n: Int) -> (agree: Double, cos: Double) {
    var same = 0, tot = 0
    var dot = 0.0, na = 0.0, nb = 0.0
    for f in 0..<min(n, frames.count) {
        for s in 0..<4 {
            let g = gold[f * 4 + s], p = frames[f][s]
            if (p > 0.5) == (g > 0.5) { same += 1 }
            tot += 1
            dot += Double(p) * Double(g); na += Double(p) * Double(p); nb += Double(g) * Double(g)
        }
    }
    return (tot == 0 ? 0 : Double(same) / Double(tot), dot / (na.squareRoot() * nb.squareRoot() + 1e-12))
}

private func secs(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock().now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
