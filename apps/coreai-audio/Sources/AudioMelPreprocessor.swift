// AudioMelPreprocessor.swift — Whisper-large-v3 log-mel in Accelerate, matching transformers'
// WhisperFeatureExtractor (the Qwen2.5-Omni audio tower front end) so a 16 kHz waveform becomes the
// encoder's input_features. Recipe = reflect-pad n_fft/2 + Hann(400) window per hop 160 + real DFT
// (cos/sin MATMUL, since n_fft=400 isn't a power of two) -> 201 power bins -> mel_filters^T @ power
// -> log10(max(.,1e-10)); max(., globalMax-8); (.+4)/4. Gated bit-exact vs python (cos 1.0).

import Accelerate
import Foundation

struct AudioMelPreprocessor: Sendable {
    let nFFT: Int, hop: Int, nMels: Int, nFreq: Int
    private let window: [Float]
    private let cosMat: [Float]      // [nFreq, nFFT]
    private let sinMat: [Float]      // [nFreq, nFFT]
    private let melFiltersT: [Float] // [nMels, nFreq]

    /// `melFilters` = the HF extractor's mel_filters, row-major [nFreq, nMels] (e.g. [201,128]).
    init(melFilters: [Float], nFFT: Int = 400, hop: Int = 160, nMels: Int = 128) {
        precondition(melFilters.count == (nFFT / 2 + 1) * nMels, "mel_filters shape mismatch")
        self.nFFT = nFFT; self.hop = hop; self.nMels = nMels
        let nFreq = nFFT / 2 + 1
        self.nFreq = nFreq
        var win = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT { win[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT)) }
        self.window = win
        var c = [Float](repeating: 0, count: nFreq * nFFT)
        var s = [Float](repeating: 0, count: nFreq * nFFT)
        for k in 0..<nFreq {
            for n in 0..<nFFT {
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                c[k * nFFT + n] = cos(a); s[k * nFFT + n] = sin(a)
            }
        }
        self.cosMat = c; self.sinMat = s
        var mt = [Float](repeating: 0, count: nMels * nFreq)
        for f in 0..<nFreq {
            for m in 0..<nMels { mt[m * nFreq + f] = melFilters[f * nMels + m] }
        }
        self.melFiltersT = mt
    }

    /// The Qwen2.5-Omni extractor, built from the app's bundled mel filterbank resource.
    static func qwen2_5Omni() throws -> AudioMelPreprocessor {
        guard let url = Bundle.main.url(forResource: "mel_filters", withExtension: "f32") else {
            throw NSError(domain: "AudioMel", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "mel_filters.f32 resource missing"])
        }
        let filters = try Data(contentsOf: url).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        return AudioMelPreprocessor(melFilters: filters)
    }

    /// Log-mel for a 16 kHz mono waveform: row-major [nMels, frames], frames = samples/hop.
    func logMel(_ samples: [Float]) -> (mel: [Float], frames: Int) {
        let frames = samples.count / hop
        guard frames > 0 else { return ([], 0) }
        let pad = nFFT / 2
        let padded = reflectPad(samples, pad: pad)
        var win = [Float](repeating: 0, count: nFFT * frames)
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { win[n * frames + t] = padded[base + n] * window[n] }
        }
        var re = [Float](repeating: 0, count: nFreq * frames)
        var im = [Float](repeating: 0, count: nFreq * frames)
        matmul(cosMat, win, &re, m: nFreq, n: frames, k: nFFT)
        matmul(sinMat, win, &im, m: nFreq, n: frames, k: nFFT)
        let count = vDSP_Length(nFreq * frames)
        vDSP_vsq(re, 1, &re, 1, count)
        vDSP_vsq(im, 1, &im, 1, count)
        var power = [Float](repeating: 0, count: nFreq * frames)
        vDSP_vadd(re, 1, im, 1, &power, 1, count)
        var mel = [Float](repeating: 0, count: nMels * frames)
        matmul(melFiltersT, power, &mel, m: nMels, n: frames, k: nFreq)
        let melCount = vDSP_Length(mel.count)
        var floorV: Float = 1e-10
        vDSP_vthr(mel, 1, &floorV, &mel, 1, melCount)   // max(mel, 1e-10)
        var n32 = Int32(mel.count)
        vvlog10f(&mel, mel, &n32)
        var globalMax: Float = -.greatestFiniteMagnitude
        vDSP_maxv(mel, 1, &globalMax, melCount)
        var clampLow = globalMax - 8.0
        vDSP_vthr(mel, 1, &clampLow, &mel, 1, melCount) // max(mel, globalMax-8)
        var add: Float = 4.0, div: Float = 4.0
        vDSP_vsadd(mel, 1, &add, &mel, 1, melCount)
        vDSP_vsdiv(mel, 1, &div, &mel, 1, melCount)
        return (mel, frames)
    }

    private func matmul(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int) {
        vDSP_mmul(a, 1, b, 1, &c, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }

    private func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }
        for i in 0..<n { out[pad + i] = x[i] }
        for i in 0..<pad { out[pad + n + i] = x[n - 2 - i] }
        return out
    }
}
