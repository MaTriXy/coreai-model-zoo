// ActionModel — V-JEPA 2 (ViT-L, SSv2) action recognition on Core AI. One stateless graph via the
// kit's GraphModel: pixel_values_videos [1,16,3,256,256] (RGB 0..1, ImageNet norm) -> logits [1,174].
//
// Assets: `VJEPAAssets` root (dev symlink -> conversion/vjepa2/ship_macos) holds the bundle +
// labels.json. macOS uses `.aimodel`; device uses the AOT `.h18p.aimodelc`.
import CoreAIKitVision
import CoreGraphics
import Foundation

enum VJEPAAssets {
    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("VJEPAAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VJEPAAssets")
        #endif
    }
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("labels.json").path) ? p : nil
    }
}

struct ActionPrediction: Identifiable {
    let id: Int
    let label: String
    let prob: Float
}

final class ActionEngine: @unchecked Sendable {
    static let frames = 16, side = 256
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    private let model: GraphModel
    private let labels: [String]

    init(root: URL) async throws {
        #if os(macOS)
        let bundle = root.appendingPathComponent("vjepa2_ssv2_fp16.aimodel")
        #else
        let bundle = root.appendingPathComponent("vjepa2_ssv2_fp16.h18p.aimodelc")
        #endif
        model = try await GraphModel(contentsOf: bundle, computeUnits: .gpu)
        let data = try Data(contentsOf: root.appendingPathComponent("labels.json"))
        let map = try JSONDecoder().decode([String: String].self, from: data)
        labels = (0..<map.count).map { map[String($0)] ?? "class \($0)" }
    }

    /// tensor: [1,16,3,256,256] row-major floats, already normalized.
    func classify(_ tensor: [Float], topK: Int = 3) async throws -> [ActionPrediction] {
        let out = try await model.run([
            "pixel_values_videos": .float32(tensor, shape: [1, Self.frames, 3, Self.side, Self.side])
        ])
        let logits = out["logits"]!.floats()
        // softmax
        let mx = logits.max() ?? 0
        let exps = logits.map { expf($0 - mx) }
        let sum = exps.reduce(0, +)
        return exps.enumerated()
            .map { ActionPrediction(id: $0.offset, label: labels[$0.offset], prob: $0.element / sum) }
            .sorted { $0.prob > $1.prob }
            .prefix(topK).map { $0 }
    }

    /// 16 CGImages (any size) -> normalized [1,16,3,256,256] tensor. Center-crop + 256 resize.
    nonisolated static func tensor(from images: [CGImage]) -> [Float] {
        let S = side, plane = S * S
        var out = [Float](repeating: 0, count: frames * 3 * plane)
        var rgba = [UInt8](repeating: 0, count: plane * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        for (t, img) in images.prefix(frames).enumerated() {
            rgba.withUnsafeMutableBytes { buf in
                guard let ctx = CGContext(data: buf.baseAddress, width: S, height: S,
                                          bitsPerComponent: 8, bytesPerRow: S * 4, space: cs,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
                // aspect-fill so the shorter side maps to 256 (center crop)
                let w = CGFloat(img.width), h = CGFloat(img.height)
                let scale = CGFloat(S) / min(w, h)
                let dw = w * scale, dh = h * scale
                ctx.interpolationQuality = .medium
                ctx.draw(img, in: CGRect(x: (CGFloat(S) - dw) / 2, y: (CGFloat(S) - dh) / 2, width: dw, height: dh))
            }
            let base = t * 3 * plane
            for c in 0..<3 {
                let m = mean[c], sd = std[c], cbase = base + c * plane
                for p in 0..<plane {
                    out[cbase + p] = (Float(rgba[p * 4 + c]) / 255.0 - m) / sd
                }
            }
        }
        return out
    }
}
