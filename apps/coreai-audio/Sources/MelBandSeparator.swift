// MelBandSeparator — Mel-Band RoFormer (Kim Vocal, MIT) vocal / instrumental source
// separation on Core AI. zoo's first source-separation model.
//
// The exported graph folds STFT + the RoFormer core + iSTFT into one bundle
// (frames[1,2,nfr,N] -> recon[1,2,nfr,N]); the host only reflect-pads, slices
// frames, and overlap-adds — NO FFT. Two overlap-add levels, both mirroring
// conversion/melband_roformer/{export_core2,pipeline_engine}.py exactly:
//   (a) inside an 8 s chunk: STFT-frame overlap-add (stride hop) / sum(win^2) -> chunk audio
//   (b) across the song: chunk overlap-add (stride C/num_overlap) with fade windows
// instrumental = mix - vocals.
import CoreAIKitVision
import Foundation

actor MelBandSeparator {
    static let nFFT = 2048, hop = 441, pad = 1024
    static let C = 352_800                 // 8 s @ 44.1 kHz (chunk_size)
    static let numOverlap = 2
    static let sr = 44_100
    static var nFrames: Int { 1 + (C + 2 * pad - nFFT) / hop }   // 801

    private let model: GraphModel
    private let window: [Float]             // Hann periodic [nFFT]
    private let wsum: [Float]               // sum(win^2) overlap over a padded chunk

    init(model url: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        model = try await GraphModel(contentsOf: url, computeUnits: computeUnits)
        var w = [Float](repeating: 0, count: Self.nFFT)
        for n in 0..<Self.nFFT { w[n] = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(Self.nFFT)) }
        window = w
        let total = Self.nFFT + Self.hop * (Self.nFrames - 1)
        var s = [Float](repeating: 0, count: total)
        for i in 0..<Self.nFrames { for n in 0..<Self.nFFT { s[i * Self.hop + n] += w[n] * w[n] } }
        for k in 0..<total { s[k] = max(s[k], 1e-8) }
        wsum = s
    }

    /// numpy 'reflect' padding (reflect_101: edge not repeated).
    private static func reflectPad(_ x: [Float], _ p: Int) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * p)
        for j in 0..<p { out[j] = x[p - j] }                 // left
        for k in 0..<n { out[p + k] = x[k] }
        for j in 0..<p { out[p + n + j] = x[n - 2 - j] }     // right
        return out
    }

    /// One exactly-C-sample chunk -> vocals [2][C].  pad+frame -> graph -> frame overlap-add.
    /// Public: the self-test drives this directly (matches golden_vocals.f32).
    func separateChunk(_ chunk: [[Float]]) async throws -> [[Float]] {
        let N = Self.nFFT, H = Self.hop, F = Self.nFrames
        // reflect-pad both channels, slice frames -> flat [1,2,F,N]
        var flat = [Float](repeating: 0, count: 2 * F * N)
        for c in 0..<2 {
            let xp = Self.reflectPad(chunk[c], Self.pad)
            for i in 0..<F {
                let base = i * H
                let dst = (c * F + i) * N
                for n in 0..<N { flat[dst + n] = xp[base + n] }
            }
        }
        let out = try await model.run(["frames": .float32(flat, shape: [1, 2, F, N])])
        let recon = out["recon"]!.floats()                   // [2*F*N]
        // frame overlap-add -> / wsum -> trim pad
        let total = N + H * (F - 1)
        var acc = [[Float]](repeating: [Float](repeating: 0, count: total), count: 2)
        for c in 0..<2 {
            for i in 0..<F {
                let base = i * H, src = (c * F + i) * N
                for n in 0..<N { acc[c][base + n] += recon[src + n] }
            }
            for k in 0..<total { acc[c][k] /= wsum[k] }
        }
        return [Array(acc[0][Self.pad..<Self.pad + Self.C]),
                Array(acc[1][Self.pad..<Self.pad + Self.C])]
    }

    /// Full mix [2][T] -> (vocals, instrumental), each [2][T]. Chunk overlap-add (fade).
    func separate(_ mix: [[Float]], progress: (@Sendable (Double) -> Void)? = nil) async throws -> (vocals: [[Float]], instrumental: [[Float]]) {
        let C = Self.C, step = C / Self.numOverlap, fade = C / 10
        let T = mix[0].count
        let border = C - step
        let mixp = [Self.reflectPad(mix[0], border), Self.reflectPad(mix[1], border)]
        let total = mixp[0].count

        // fade window (ones with fade-in/out at the ends); linspace(0,1,fade) = k/(fade-1)
        var win = [Float](repeating: 1, count: C)
        for k in 0..<fade { let v = Float(k) / Float(fade - 1); win[k] = v; win[C - 1 - k] = v }

        var result = [[Float]](repeating: [Float](repeating: 0, count: total), count: 2)
        var counter = [Float](repeating: 0, count: total)
        var i = 0
        let nChunks = max(1, (total + step - 1) / step)
        var done = 0
        while i < total {
            var part = [[Float]](repeating: [Float](repeating: 0, count: C), count: 2)
            let L = min(C, total - i)
            for c in 0..<2 { for k in 0..<L { part[c][k] = mixp[c][i + k] } }
            if L < C {   // reflect-pad the short tail to a full chunk (numpy 'reflect')
                for c in 0..<2 {
                    for k in L..<C {
                        let src = 2 * L - 2 - k
                        part[c][k] = src >= 0 ? part[c][src] : 0
                    }
                }
            }
            let voc = try await separateChunk(part)
            var w = win
            if i == 0 { for k in 0..<fade { w[k] = 1 } }
            else if i + C >= total { for k in 0..<fade { w[C - 1 - k] = 1 } }
            for c in 0..<2 { for k in 0..<L { result[c][i + k] += voc[c][k] * w[k] } }
            for k in 0..<L { counter[i + k] += w[k] }
            i += step; done += 1
            progress?(min(1, Double(done) / Double(nChunks)))
        }
        var vocals = [[Float]](repeating: [Float](repeating: 0, count: T), count: 2)
        var inst = [[Float]](repeating: [Float](repeating: 0, count: T), count: 2)
        for c in 0..<2 {
            for k in 0..<T {
                let v = result[c][border + k] / max(counter[border + k], 1e-8)
                vocals[c][k] = v
                inst[c][k] = mix[c][k] - v
            }
        }
        return (vocals, inst)
    }
}
