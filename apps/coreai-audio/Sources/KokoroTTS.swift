// KokoroTTS — Kokoro-82M (StyleTTS2 + iSTFTNet) text-to-speech on Core AI.
//
// Drives the three exported bundles (predictor / prosody / vocoder) through
// CoreAIKit's GraphModel on the CPU compute unit, with the host steps in Swift:
// tokenize -> predictor -> alignment -> prosody -> compute_har -> vocoder -> trim.
// G2P is host-side; this demo uses phrases phonemized ahead of time (see
// demo_phrases.json) so it carries no MLX/espeak dependency. The host DSP
// (alignment + compute_har) mirrors conversion/export_kokoro.py exactly.
import AVFoundation
import CoreAIKitVision
import Foundation

actor KokoroTTS {
    static let TB = 128          // token bucket
    static let LB = 512          // frame bucket
    static let UP = 300          // f0 upsample (prod(upsample_rates)*hop)
    static let NFFT = 20, HOP = 5, FREQ = 11, HARM = 9
    static let SR: Float = 24000

    private let predictor: GraphModel
    private let prosody: GraphModel
    private let vocoder: GraphModel
    private let vocab: [String: Int]
    private let llW: [Float]      // l_linear weight[9] + bias[1]
    private var voices: [String: [Float]] = [:]   // name -> [N*256]

    // STFT DFT basis (cos/sin * periodic Hann), built once.
    private let basisR: [Float]   // [FREQ*NFFT]
    private let basisI: [Float]

    init(predictor pURL: URL, prosody prURL: URL, vocoder vURL: URL,
         vocab vocabURL: URL, lLinear lURL: URL) async throws {
        predictor = try await GraphModel(contentsOf: pURL, computeUnits: .cpu)
        prosody = try await GraphModel(contentsOf: prURL, computeUnits: .cpu)
        vocoder = try await GraphModel(contentsOf: vURL, computeUnits: .cpu)
        let vdata = try Data(contentsOf: vocabURL)
        vocab = try JSONDecoder().decode([String: Int].self, from: vdata)
        llW = Self.readFloats(lURL)

        var br = [Float](), bi = [Float]()
        br.reserveCapacity(Self.FREQ * Self.NFFT)
        bi.reserveCapacity(Self.FREQ * Self.NFFT)
        for k in 0..<Self.FREQ {
            for n in 0..<Self.NFFT {
                let hann = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(Self.NFFT))
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(Self.NFFT)
                br.append(cos(a) * hann)
                bi.append(-sin(a) * hann)
            }
        }
        basisR = br; basisI = bi
    }

    func availableVoices() -> [String] { Array(voices.keys).sorted() }

    func loadVoice(_ name: String, url: URL) {
        voices[name] = Self.readFloats(url)         // [N, 256] flattened
    }

    /// Synthesize from already-tokenized phoneme ids (incl. the [0, …, 0] bounds).
    func synthesize(ids rawIds: [Int], voice: String) async throws -> [Float] {
        let T = rawIds.count
        guard T <= Self.TB else { throw TTSError.tooLong(T) }
        guard let pack = voices[voice] else { throw TTSError.noVoice(voice) }
        let refS = Array(pack[(T - 1) * 256 ..< T * 256])       // ref_s = pack[len-1]

        // pad ids to the token bucket
        var ids = rawIds.map { Int32($0) } + [Int32](repeating: 0, count: Self.TB - T)
        var attn = [Float](repeating: 1, count: T) + [Float](repeating: 0, count: Self.TB - T)

        let o1 = try await predictor.run([
            "input_ids": .int32(ids, shape: [1, Self.TB]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "attn_mask": .float32(attn, shape: [1, Self.TB]),
        ])
        let duration = o1["duration"]!.floats()

        let (aln, frameMask, L) = buildAlignment(duration: Array(duration[0..<T]), T: T)

        let o2 = try await prosody.run([
            "d": o1["d"]!, "t_en": o1["t_en"]!,
            "aln": .float32(aln, shape: [1, Self.TB, Self.LB]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "frame_mask": .float32(frameMask, shape: [1, Self.LB]),
        ])
        let F0 = o2["F0"]!.floats()                              // [2*LB]
        let (har, frames) = computeHar(F0: F0)

        let o3 = try await vocoder.run([
            "asr": o2["asr"]!, "F0": o2["F0"]!, "N": o2["N"]!,
            "har": .float32(har, shape: [1, 2 * Self.FREQ, frames]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "frame_mask": .float32(frameMask, shape: [1, Self.LB]),
        ])
        let audio = o3["audio"]!.floats()
        return Array(audio[0 ..< min(L * 600, audio.count)])     // trim to real frames
    }

    // host step 1: duration -> one-hot alignment + frame mask (bucketed)
    private func buildAlignment(duration: [Float], T: Int) -> ([Float], [Float], Int) {
        var predDur = [Int](repeating: 1, count: T)
        var L = 0
        for i in 0..<T {
            predDur[i] = max(1, Int((duration[i]).rounded()))
            L += predDur[i]
        }
        var aln = [Float](repeating: 0, count: Self.TB * Self.LB)
        var frame = 0
        for i in 0..<T {
            for _ in 0..<predDur[i] where frame < Self.LB {
                aln[i * Self.LB + frame] = 1                     // aln[token i, frame]
                frame += 1
            }
        }
        var frameMask = [Float](repeating: 0, count: Self.LB)
        for f in 0..<min(L, Self.LB) { frameMask[f] = 1 }
        return (aln, frameMask, L)
    }

    // host step 2: hn-nsf source -> STFT (mag, phase). Mirrors compute_har.
    private func computeHar(F0: [Float]) -> ([Float], Int) {
        let twoL = F0.count                 // 2*LB
        let bigL = twoL * Self.UP
        // f0_upsamp: repeat each value UP times (nearest)
        // SineGen per harmonic: rad -> downsample(1/UP) -> cumsum -> upsample(UP) -> sin
        var harSource = [Float](repeating: 0, count: bigL)
        var radDS = [Float](repeating: 0, count: twoL)
        var phase = [Float](repeating: 0, count: twoL)
        for h in 1...Self.HARM {
            // rad on the bigL grid is f0_up*h/SR; its 1/UP downsample averages each
            // block of UP constant samples -> ~= F0[j]*h/SR (linear interp of a step).
            for j in 0..<twoL { radDS[j] = F0[j] * Float(h) / Self.SR }
            // cumsum * 2pi
            var acc: Float = 0
            for j in 0..<twoL { acc += radDS[j]; phase[j] = acc * 2 * Float.pi }
            // upsample (linear, align_corners=false) of phase*UP back to bigL, then sin
            let weight = llW[h - 1]
            for i in 0..<bigL {
                let s = max(0, min(Float(twoL - 1), (Float(i) + 0.5) * Float(twoL) / Float(bigL) - 0.5))
                let lo = Int(s.rounded(.down)); let hi = min(lo + 1, twoL - 1); let fr = s - Float(lo)
                let ph = (phase[lo] * (1 - fr) + phase[hi] * fr) * Float(Self.UP)
                let f0u = F0[min(twoL - 1, i / Self.UP)]                       // f0_upsamp value
                let uv: Float = f0u > 10 ? 1 : 0
                harSource[i] += sin(ph) * 0.1 * uv * weight                    // sine_amp * uv * l_linear[h-1]
            }
        }
        // l_linear bias + tanh
        let bias = llW[Self.HARM]
        for i in 0..<bigL { harSource[i] = tanh(harSource[i] + bias) }

        // STFT: replicate pad n_fft/2, stride HOP, DFT basis -> mag, phase
        let pad = Self.NFFT / 2
        let total = bigL + 2 * pad
        let frames = (total - Self.NFFT) / Self.HOP + 1
        var har = [Float](repeating: 0, count: 2 * Self.FREQ * frames)
        func sample(_ idx: Int) -> Float {                     // replicate (edge) pad
            if idx < pad { return harSource[0] }
            if idx >= pad + bigL { return harSource[bigL - 1] }
            return harSource[idx - pad]
        }
        for fi in 0..<frames {
            let base = fi * Self.HOP
            for k in 0..<Self.FREQ {
                var re: Float = 0, im: Float = 0
                let bk = k * Self.NFFT
                for n in 0..<Self.NFFT {
                    let v = sample(base + n)
                    re += basisR[bk + n] * v
                    im += basisI[bk + n] * v
                }
                let mag = (re * re + im * im + 1e-14).squareRoot()
                har[k * frames + fi] = mag
                har[(Self.FREQ + k) * frames + fi] = atan2(im, re)
            }
        }
        return (har, frames)
    }

    private static func readFloats(_ url: URL) -> [Float] {
        guard let d = try? Data(contentsOf: url) else { return [] }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    enum TTSError: Error { case tooLong(Int), noVoice(String) }
}

// 24 kHz mono playback.
@MainActor
final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var started = false

    func play(_ samples: [Float], sampleRate: Double = 24000) {
        guard let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                      channels: 1, interleaved: false) else { return }
        if !started {
            #if os(iOS)
            try? AVAudioSession.sharedInstance().setCategory(.playback)
            try? AVAudioSession.sharedInstance().setActive(true)
            #endif
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: fmt)
            try? engine.start()
            started = true
        }
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buf.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        player.scheduleBuffer(buf, completionHandler: nil)
        player.play()
    }
}
