// WhisperMel — log-mel spectrogram frontend matching OpenAI Whisper's feature extractor
// (n_fft 400, hop 160, 128 mel bins, 16 kHz, 30 s window → [1, 128, 3000]).
//
// Whisper's n_fft is 400, which isn't a vDSP-FFT-friendly length, so the 400-point DFT is
// computed as a matmul against a precomputed cos/sin basis ([400 × 201]). The mel filterbank
// ([201 × 128], shipped alongside the model as mel_filters_128.npy) is applied as a second
// matmul. Both use BLAS (cblas_sgemm); element-wise log/normalize use vDSP.

import Accelerate
import Foundation

enum WhisperMel {
    static let sampleRate = 16_000
    static let nFFT = 400
    static let hop = 160
    static let nMels = 128
    static let nFreq = nFFT / 2 + 1          // 201
    static let chunkFrames = 3000            // 30 s
    static let nSamples = chunkFrames * hop  // 480000

    /// Precomputed once: Hann window, DFT cos/sin basis [nFFT × nFreq].
    private static let hann: [Float] = (0..<nFFT).map {
        0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(nFFT))
    }
    private static let dftCos: [Float] = basis(sign: 1)
    private static let dftSin: [Float] = basis(sign: -1)  // -sin for the imaginary part

    private static func basis(sign: Float) -> [Float] {
        // [nFFT × nFreq] row-major: row t (time sample), col k (freq bin).
        var m = [Float](repeating: 0, count: nFFT * nFreq)
        for t in 0..<nFFT {
            for k in 0..<nFreq {
                let ang = 2 * Float.pi * Float(t) * Float(k) / Float(nFFT)
                m[t * nFreq + k] = sign > 0 ? cosf(ang) : -sinf(ang)
            }
        }
        return m
    }

    /// 16 kHz mono samples → log-mel `[1, 128, 3000]` (mel-major, row = mel, col = frame),
    /// flattened row-major as the model's `input_features` expects.
    static func logMel(samples: [Float], melFilters: [Float]) -> [Float] {
        // Pad / trim to exactly 30 s, then reflect-pad nFFT/2 on each side (torch center=True).
        var x = samples
        if x.count < nSamples { x.append(contentsOf: repeatElement(0, count: nSamples - x.count)) }
        else if x.count > nSamples { x = Array(x[0..<nSamples]) }

        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: x.count + 2 * pad)
        for i in 0..<pad { padded[i] = x[pad - i] }                       // reflect left
        for i in 0..<x.count { padded[pad + i] = x[i] }
        for i in 0..<pad { padded[pad + x.count + i] = x[x.count - 2 - i] } // reflect right

        // Frame: hop 160, length 400, then drop the last frame → 3000 frames.
        let totalFrames = 1 + (padded.count - nFFT) / hop
        let frames = min(totalFrames - 1, chunkFrames)
        // Windowed frames matrix [frames × nFFT], row-major.
        var win = [Float](repeating: 0, count: frames * nFFT)
        padded.withUnsafeBufferPointer { p in
            for f in 0..<frames {
                let off = f * hop
                vDSP_vmul(p.baseAddress! + off, 1, hann, 1, &win[f * nFFT], 1, vDSP_Length(nFFT))
            }
        }

        // DFT via matmul: real/imag = win[frames × nFFT] @ basis[nFFT × nFreq] → [frames × nFreq].
        var real = [Float](repeating: 0, count: frames * nFreq)
        var imag = [Float](repeating: 0, count: frames * nFreq)
        sgemm(a: win, b: dftCos, c: &real, m: frames, k: nFFT, n: nFreq)
        sgemm(a: win, b: dftSin, c: &imag, m: frames, k: nFFT, n: nFreq)

        // power = real^2 + imag^2  → [frames × nFreq]
        var power = [Float](repeating: 0, count: frames * nFreq)
        vDSP_vsq(real, 1, &real, 1, vDSP_Length(real.count))
        vDSP_vsq(imag, 1, &imag, 1, vDSP_Length(imag.count))
        vDSP_vadd(real, 1, imag, 1, &power, 1, vDSP_Length(power.count))

        // mel = power[frames × nFreq] @ melFilters[nFreq × nMels] → [frames × nMels]
        var mel = [Float](repeating: 0, count: frames * nMels)
        sgemm(a: power, b: melFilters, c: &mel, m: frames, k: nFreq, n: nMels)

        // log10(clamp(mel, 1e-10)); clamp to max-8; (x+4)/4
        var floor = Float(1e-10)
        vDSP_vthr(mel, 1, &floor, &mel, 1, vDSP_Length(mel.count))
        var n = Int32(mel.count)
        vvlog10f(&mel, mel, &n)
        var maxv: Float = 0
        vDSP_maxv(mel, 1, &maxv, vDSP_Length(mel.count))
        var lo = maxv - 8.0
        vDSP_vthr(mel, 1, &lo, &mel, 1, vDSP_Length(mel.count))
        var add: Float = 4.0, div: Float = 4.0
        vDSP_vsadd(mel, 1, &add, &mel, 1, vDSP_Length(mel.count))
        vDSP_vsdiv(mel, 1, &div, &mel, 1, vDSP_Length(mel.count))

        // Transpose [frames × nMels] → mel-major [nMels × chunkFrames], zero-padded to 3000.
        var out = [Float](repeating: 0, count: nMels * chunkFrames)
        for m in 0..<nMels {
            for f in 0..<frames {
                out[m * chunkFrames + f] = mel[f * nMels + m]
            }
        }
        return out
    }

    /// C = A[m×k] · B[k×n], all row-major Float (vDSP_mmul: C[M×N] = A[M×P]·B[P×N]).
    private static func sgemm(a: [Float], b: [Float], c: inout [Float], m: Int, k: Int, n: Int) {
        vDSP_mmul(a, 1, b, 1, &c, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }

    // MARK: - mel_filters_128.npy loader

    /// Parse a NumPy `.npy` (v1, C-order, little-endian float32) into a flat `[Float]`.
    /// Used for `mel_filters_128.npy` (shape [201, 128]).
    static func loadNpyFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 10, data[0] == 0x93,
              data[1...5] == Data("NUMPY".utf8) else {
            throw NSError(domain: "WhisperMel", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "not a .npy file"])
        }
        let headerLen = Int(data[8]) | (Int(data[9]) << 8)
        let dataStart = 10 + headerLen
        let header = String(decoding: data[10..<dataStart], as: UTF8.self)
        guard header.contains("'<f4'"), !header.contains("True") else {
            throw NSError(domain: "WhisperMel", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "expected C-order <f4 npy"])
        }
        let floatBytes = data[dataStart...]
        return floatBytes.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }
}
