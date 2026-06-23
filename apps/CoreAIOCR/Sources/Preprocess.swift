// Preprocess.swift — CGImage -> [3,640,640] fp16, the Unlimited-OCR Base-mode transform.
//
// Mirrors PIL `ImageOps.pad(image, (640,640), color=mean*255)` + normalize(mean=std=0.5):
// aspect-preserving resize to fit 640x640, center, pad the remainder with gray (127.5),
// then (x/255 - 0.5)/0.5 = x/127.5 - 1. Output is channel-major (R plane, G plane, B plane).
//
// VERIFY ON DEVICE against out/_vision_oracle (sam_in0) once: the engine pipeline is exact;
// any small mismatch here would be in this resampling, not the model.

import CoreGraphics
import Foundation

enum Preprocess {
    static func toCHW640(_ image: CGImage) -> [Float16] {
        let S = OCRSpec.imageSide
        let gray: UInt8 = 128  // mean*255 = 127.5 ~ 128 pad color

        // aspect-preserving fit
        let scale = min(Double(S) / Double(image.width), Double(S) / Double(image.height))
        let dw = Int((Double(image.width) * scale).rounded())
        let dh = Int((Double(image.height) * scale).rounded())
        let ox = (S - dw) / 2, oy = (S - dh) / 2

        var rgba = [UInt8](repeating: gray, count: S * S * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: S, height: S, bitsPerComponent: 8,
                bytesPerRow: S * 4, space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            // CG origin is bottom-left; oy from the top maps to (S-dh-oy) at the bottom.
            ctx.draw(image, in: CGRect(x: ox, y: S - dh - oy, width: dw, height: dh))
        }

        var out = [Float16](repeating: 0, count: 3 * S * S)
        let plane = S * S
        for y in 0..<S {
            for x in 0..<S {
                let p = (y * S + x) * 4
                let idx = y * S + x
                out[idx]             = Float16(Float(rgba[p])     / 127.5 - 1)  // R
                out[plane + idx]     = Float16(Float(rgba[p + 1]) / 127.5 - 1)  // G
                out[2 * plane + idx] = Float16(Float(rgba[p + 2]) / 127.5 - 1)  // B
            }
        }
        return out
    }
}
