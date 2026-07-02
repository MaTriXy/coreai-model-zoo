// Headless self-test (ACTION_SELFTEST=1): load V-JEPA 2, feed synthetic up/down motion (a red
// square moving across gray, 16 frames), check the direction-sensitive labels flip. Mirrors
// conversion/vjepa2/gate_semantic.py so app-side preprocessing is validated too.
import CoreGraphics
import Foundation

func runActionSelfTest() async {
    func log(_ s: String) { print("[ACTION] \(s)"); fflush(stdout) }
    guard let root = VJEPAAssets.root else {
        log("FAIL: assets not found at \(VJEPAAssets.location.path)"); return
    }
    do {
        let t0 = ContinuousClock.now
        let engine = try await ActionEngine(root: root)
        log("loaded in \(secs(since: t0))s")
        var pass = true
        for direction in ["up", "down"] {
            let frames = syntheticFrames(direction: direction)
            let tensor = ActionEngine.tensor(from: frames)
            let g0 = ContinuousClock.now
            let preds = try await engine.classify(tensor, topK: 5)
            log("\(direction): \(secs(since: g0))s  top: " + preds.prefix(3).map { "\($0.label) \(String(format: "%.2f", $0.prob))" }.joined(separator: " | "))
            let want = direction == "up" ? "Moving [something] up" : "Moving [something] down"
            if !preds.contains(where: { $0.label == want }) { pass = false; log("MISSING \(want) in top5") }
        }
        log(pass ? "PASS" : "FAIL")
    } catch { log("FAIL: \(error)") }
}

private func secs(since t: ContinuousClock.Instant) -> String {
    let d = ContinuousClock.now - t
    return String(format: "%.2f", Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
}

private func syntheticFrames(direction: String) -> [CGImage] {
    let S = 256, sq = 48
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    return (0..<ActionEngine.frames).map { t in
        let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8, bytesPerRow: S * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(srgbRed: 0.35, green: 0.35, blue: 0.35, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
        let p = CGFloat(t) / CGFloat(ActionEngine.frames - 1)
        // CG origin is bottom-left: visual "up" = increasing y
        let cy = direction == "up" ? (0.2 + 0.6 * p) : (0.8 - 0.6 * p)
        ctx.setFillColor(CGColor(srgbRed: 0.9, green: 0.2, blue: 0.15, alpha: 1))
        ctx.fill(CGRect(x: S / 2 - sq / 2, y: Int(cy * CGFloat(S)) - sq / 2, width: sq, height: sq))
        return ctx.makeImage()!
    }
}
