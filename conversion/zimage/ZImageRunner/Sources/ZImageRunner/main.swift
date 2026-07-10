// Drives the app's ZImagePipeline from the command line so the Swift host loop can be
// gated against the Python reference (pipeline_engine.py) — same bundle, same prompt.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func opt(_ k: String) -> String? {
    guard let i = CommandLine.arguments.firstIndex(of: k), i + 1 < CommandLine.arguments.count
    else { return nil }
    return CommandLine.arguments[i + 1]
}

let bundle = URL(fileURLWithPath: opt("--bundle") ?? NSHomeDirectory() + "/Documents/Z-Image-Turbo")
let side = Int(opt("--side") ?? "512") ?? 512
let steps = Int(opt("--steps") ?? "8") ?? 8
let guidance = Float(opt("--guidance") ?? "1.0") ?? 1.0
let seed = UInt64(opt("--seed") ?? "1234") ?? 1234
let prompt = opt("--prompt") ?? "a red apple on a wooden table, studio lighting"
let negative = opt("--negative") ?? ""
let outPath = opt("--out") ?? "/tmp/zimage_swift.png"

// Replay the Python reference's noise so the two engines can be compared pixel-for-pixel.
var initial: [Float]? = nil
if let np = opt("--noise") {
    let d = try Data(contentsOf: URL(fileURLWithPath: np))
    initial = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    print("[runner] loaded \(initial!.count) noise values from \(np)")
}

let pipe = ZImagePipeline(dir: bundle)
let t0 = Date()
try await pipe.load()
print(String(format: "[runner] loaded in %.1fs (sides %@)", Date().timeIntervalSince(t0),
             pipe.availableSides.map(String.init).joined(separator: ",")))

let t1 = Date()
let cg = try await pipe.generate(prompt: prompt, negativePrompt: negative, side: side,
                                 steps: steps, guidance: guidance, seed: seed,
                                 initialLatent: initial) { s, n in
    if s > 0 { print("[runner] step \(s)/\(n)") }
    return true
}
print(String(format: "[runner] generated %dx%d in %.1fs", cg.width, cg.height,
             Date().timeIntervalSince(t1)))

let url = URL(fileURLWithPath: outPath)
guard let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
else { fatalError("cannot write \(outPath)") }
CGImageDestinationAddImage(dst, cg, nil)
CGImageDestinationFinalize(dst)
print("[runner] wrote \(outPath)")
