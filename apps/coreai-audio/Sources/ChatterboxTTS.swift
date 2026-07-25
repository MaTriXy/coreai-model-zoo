// ChatterboxTTS — Resemble AI Chatterbox zero-shot voice-cloning TTS on Core AI.
// Text -> speech, entirely on device. This is the productionized form of the pipeline proven
// end-to-end on iPhone 17 Pro in PipelinedBench (`chatterboxSpeakBench`); see
// knowledge/chatterbox-port.md for the gates and findings.
//
//   text tokens
//     -> T3 AR loop (int8 stateful graph, CFG 0.5 = 2 KV caches, temp/rep-pen/top-p sampling)
//     -> speech tokens (natural stop)
//     -> S3Gen: input-embedding + encoder graph (bucket 256) + CFM Euler-10 (estimator fp16,
//        bucket 512, CFG 0.7, real-region mask)  -> mel target
//     -> HiFT: host m_source(HnNSF) + host STFT(16) + trunk graph + host iSTFT(16)  -> 24 kHz wav
//
// The T3 is stateful (KV) so it uses the low-level InferenceFunction + MutableViews path (the kit's
// GraphModel rejects stateful graphs); the S3Gen/vocoder nets are stateless. inputs_embeds + KV are
// Float16. Graphs are static-shaped + bucketed (dynamic export is blocked by the encoder 2x-upsample
// and HiFT s_stft coupling — see the port notes).
import AVFoundation
import CoreAI
import Foundation

actor ChatterboxTTS {
    // dims / specials
    private let HID = 1024, VOCAB = 8194, EDIM = 512, NMEL = 80
    private let startSpeech = 6561, stopSpeech = 6562, codebook = 6561, lenCond = 34, promptMel = 314
    private let TOK_BUCKET = 256, MEL_BUCKET = 512, VOC_BUCKET = 256   // encoder / estimator / vocoder
    private let NFFT = 16, HOP = 4, UP = 480, HARM = 9

    // graphs
    private let t3: (InferenceFunction, InferenceFunctionDescriptor)
    private let enc: (InferenceFunction, InferenceFunctionDescriptor)
    private let est: (InferenceFunction, InferenceFunctionDescriptor)
    private let f0p: (InferenceFunction, InferenceFunctionDescriptor)
    private let hift: (InferenceFunction, InferenceFunctionDescriptor)
    // host data
    private let condPrefix, textEmb, textPos, speechEmb, speechPos: [Float]  // T3 tables
    private let s3InputEmb, s3Spk, s3PromptFeat: [Float]
    private let s3PromptTok: [Int]
    private let stftWin, llw: [Float]; private let llb: Float

    init(dir: URL) async throws {
        let opt = SpecializationOptions(preferredComputeUnitKind: .gpu)
        t3 = try await loadGraph(dir, "chatterbox_t3_decode_int8", opt)
        enc = try await loadGraph(dir, "chatterbox_s3gen_encoder_b256", opt)
        est = try await loadGraph(dir, "chatterbox_s3gen_estimator_b512", opt)
        f0p = try await loadGraph(dir, "chatterbox_f0_predictor_b256", opt)
        hift = try await loadGraph(dir, "chatterbox_s3gen_hift_trunk_b256", opt)
        condPrefix = readFloatBin(dir, "t3_cond_prefix.bin"); textEmb = readFloatBin(dir, "t3_text_emb.bin"); textPos = readFloatBin(dir, "t3_text_pos.bin")
        speechEmb = readFloatBin(dir, "t3_speech_emb.bin"); speechPos = readFloatBin(dir, "t3_speech_pos.bin")
        s3InputEmb = readFloatBin(dir, "s3_input_emb.bin"); s3Spk = readFloatBin(dir, "s3_spk.bin"); s3PromptFeat = readFloatBin(dir, "s3_prompt_feat.bin")
        s3PromptTok = readFloatBin(dir, "s3_prompt_token.bin").map { Int($0) }
        stftWin = readFloatBin(dir, "istft_window.bin"); llw = readFloatBin(dir, "msource_llinear_w.bin"); llb = readFloatBin(dir, "msource_llinear_b.bin")[0]
    }

    /// text tokens (with BOT/EOT) -> 24 kHz audio in the default cloned voice.
    func synthesize(textTokens: [Int], seed: UInt64 = 0x9E3779B97F4A7C15) async throws -> [Float] {
        let speech = try await generateSpeechTokens(textTokens: textTokens, seed: seed)
        let (mel, T) = try await speechTokensToMel(speech)
        return try await vocode(mel: mel, T: T)
    }

    // MARK: T3 — AR speech-token generation (CFG + sampling, stateful KV)

    private func generateSpeechTokens(textTokens: [Int], seed: UInt64) async throws -> [Int] {
        let (fn, d) = t3; let sn = d.stateNames
        let maxSeq = 512, kvCount = 30 * 1 * 16 * maxSeq * 64
        let z = [Float16](repeating: 0, count: kvCount)
        guard case .ndArray(let knd)? = d.stateDescriptor(of: sn[0]), case .ndArray(let vnd)? = d.stateDescriptor(of: sn[1]) else { throw Err.desc }
        func mkKV() -> (NDArray, NDArray) {
            var k = NDArray(descriptor: knd.resolvingDynamicDimensions([30, 1, 16, maxSeq, 64]))
            var v = NDArray(descriptor: vnd.resolvingDynamicDimensions([30, 1, 16, maxSeq, 64]))
            fillNDArray(&k, as: Float16.self, with: z); fillNDArray(&v, as: Float16.self, with: z); return (k, v)
        }
        var (kCc, vCc) = mkKV(); var (kCu, vCu) = mkKV()             // cond + uncond (CFG)
        // prefill: cond = cond_prefix + text(emb+pos) + start_speech; uncond zeroes the text token emb (keeps pos)
        var preC = Array(condPrefix[0 ..< lenCond * HID]); var preU = preC
        for (i, t) in textTokens.enumerated() { for c in 0..<HID { preC.append(textEmb[t * HID + c] + textPos[i * HID + c]); preU.append(textPos[i * HID + c]) } }
        for c in 0..<HID { let e = speechEmb[startSpeech * HID + c] + speechPos[c]; preC.append(e); preU.append(e) }
        let P = lenCond + textTokens.count + 1
        let cfgW: Float = 0.5
        func combine(_ a: [Float], _ b: [Float]) -> [Float] { (0..<a.count).map { a[$0] + cfgW * (a[$0] - b[$0]) } }
        var rng = seed
        let lc0 = try await t3Step(fn, d, sn, preC, P, P, &kCc, &vCc)
        let lu0 = try await t3Step(fn, d, sn, preU, P, P, &kCu, &vCu)
        var gen: [Int] = []
        var tok = Self.sample(combine(lc0, lu0), gen: gen, rng: &rng)
        var processed = P
        for k in 0..<200 {
            if tok == stopSpeech { break }
            gen.append(tok)
            if gen.count >= TOK_BUCKET - s3PromptTok.count { break }   // keep prompt+gen <= encoder bucket
            var emb = [Float](repeating: 0, count: HID)
            for c in 0..<HID { emb[c] = speechEmb[tok * HID + c] + speechPos[(k + 1) * HID + c] }
            let lc = try await t3Step(fn, d, sn, emb, 1, processed + 1, &kCc, &vCc)
            let lu = try await t3Step(fn, d, sn, emb, 1, processed + 1, &kCu, &vCu)
            processed += 1; tok = Self.sample(combine(lc, lu), gen: gen, rng: &rng)
        }
        return gen.filter { $0 < codebook }
    }

    private func t3Step(_ fn: InferenceFunction, _ d: InferenceFunctionDescriptor, _ sn: [String],
                        _ embeds: [Float], _ q: Int, _ seqLen: Int, _ kC: inout NDArray, _ vC: inout NDArray) async throws -> [Float] {
        let ie = ndF16(d, "inputs_embeds", embeds, [1, q, HID])
        let pid = ndI32(d, "position_ids", (0..<seqLen).map { Int32($0) }, [1, seqLen])
        var st = InferenceFunction.MutableViews(); st.insert(&kC, for: sn[0]); st.insert(&vC, for: sn[1])
        var lg = ndOut(d, "logits", [1, q, VOCAB])
        var ov = InferenceFunction.MutableViews(); ov.insert(&lg, for: "logits")
        _ = try await fn.run(inputs: ["inputs_embeds": ie, "position_ids": pid], states: consume st, outputViews: consume ov)
        let all = flattenAsFloat(lg); return Array(all[(q - 1) * VOCAB ..< q * VOCAB])
    }

    // MARK: S3Gen — speech tokens -> mel (encoder bucket 256 + CFM Euler-10 bucket 512, CFG 0.7)

    private func speechTokensToMel(_ speech: [Int]) async throws -> ([Float], Int) {
        let toks = s3PromptTok + speech; let N = toks.count
        guard N <= TOK_BUCKET else { throw Err.tooLong(N) }
        var xs = [Float](repeating: 0, count: TOK_BUCKET * EDIM)
        for (i, t) in toks.enumerated() { for c in 0..<EDIM { xs[i * EDIM + c] = s3InputEmb[t * EDIM + c] } }
        let (efn, ed) = enc
        let xsA = ndF32(ed, "xs", xs, [1, TOK_BUCKET, EDIM])
        let lensA = ndF32(ed, "xs_lens", [Float(N)], [1])           // xs_lens traced as Float
        var muOut = ndOut(ed, "mu", [1, MEL_BUCKET, NMEL])
        var mv = InferenceFunction.MutableViews(); mv.insert(&muOut, for: "mu")
        _ = try await efn.run(inputs: ["xs": xsA, "xs_lens": lensA], states: InferenceFunction.MutableViews(), outputViews: consume mv)
        let muHBT = flattenAsFloat(muOut); let T = MEL_BUCKET
        var mu = [Float](repeating: 0, count: NMEL * T)
        for j in 0..<T { for c in 0..<NMEL { mu[c * T + j] = muHBT[j * NMEL + c] } }
        let real = 2 * N
        var cond = [Float](repeating: 0, count: NMEL * T)
        for j in 0..<promptMel { for c in 0..<NMEL { cond[c * T + j] = s3PromptFeat[j * NMEL + c] } }
        var mask = [Float](repeating: 0, count: T); for j in 0..<real { mask[j] = 1 }
        var x = Self.gaussian(NMEL * T, seed: 1234)
        let (estFn, esd) = est; let half = NMEL * T; let cfg: Float = 0.7
        var tspan = [Float](); for i in 0...10 { let u = Float(i) / 10; tspan.append(1 - cos(u * 0.5 * Float.pi)) }
        for s in 0..<10 {
            let t = tspan[s], r = tspan[s + 1], dt = r - t
            let xin = x + x, muin = mu + [Float](repeating: 0, count: half), spin = s3Spk + [Float](repeating: 0, count: NMEL)
            let coin = cond + [Float](repeating: 0, count: half), maskin = mask + mask
            var vel = ndOut(esd, "velocity", [2, NMEL, T])
            var vv = InferenceFunction.MutableViews(); vv.insert(&vel, for: "velocity")
            _ = try await estFn.run(inputs: ["x": ndF16(esd, "x", xin, [2, NMEL, T]), "mask": ndF16(esd, "mask", maskin, [2, 1, T]),
                "mu": ndF16(esd, "mu", muin, [2, NMEL, T]), "t": ndF16(esd, "t", [t, t], [2]),
                "spks": ndF16(esd, "spks", spin, [2, NMEL]), "cond": ndF16(esd, "cond", coin, [2, NMEL, T])],
                states: InferenceFunction.MutableViews(), outputViews: consume vv)
            let v = flattenAsFloat(vel)
            for i in 0..<half where mask[i % T] > 0 { x[i] += dt * ((1 + cfg) * v[i] - cfg * v[half + i]) }
        }
        let tgt = real - promptMel
        var mel = [Float](repeating: 0, count: NMEL * tgt)
        for c in 0..<NMEL { for j in 0..<tgt { mel[c * tgt + j] = x[c * T + (promptMel + j)] } }
        return (mel, tgt)
    }

    // MARK: HiFT — mel -> wav (f0 graph + host HnNSF source + host STFT + trunk graph + host iSTFT)

    private func vocode(mel melIn: [Float], T realT: Int) async throws -> [Float] {
        let VB = VOC_BUCKET, FREQ = NFFT / 2 + 1
        guard realT <= VB else { throw Err.tooLong(realT) }
        var mel = melIn
        if realT != VB { mel = [Float](repeating: 0, count: NMEL * VB); for c in 0..<NMEL { for j in 0..<realT { mel[c * VB + j] = melIn[c * realT + j] } } }
        let T = VB
        // f0
        let (f0fn, f0d) = f0p
        let melA = ndF32(f0d, "mel", mel, [1, NMEL, T])
        var f0Out = ndOut(f0d, "f0", [1, T]); var fv = InferenceFunction.MutableViews(); fv.insert(&f0Out, for: "f0")
        _ = try await f0fn.run(inputs: ["mel": melA], states: InferenceFunction.MutableViews(), outputViews: consume fv)
        let f0 = flattenAsFloat(f0Out)
        // m_source (HnNSF) + STFT
        let L = T * UP
        var source = [Float](repeating: 0, count: L); var acc = [Float](repeating: 0, count: HARM)
        for i in 0..<L {
            let f0u = f0[min(T - 1, i / UP)]; let uv: Float = f0u > 10 ? 1 : 0; var s: Float = 0
            for h in 0..<HARM { acc[h] += f0u * Float(h + 1) / 24000.0; let fr = acc[h] - acc[h].rounded(.down); s += 0.1 * sin(2 * Float.pi * fr) * llw[h] }
            source[i] = tanh(s + llb) * uv
        }
        let pad = NFFT / 2, frames = 1 + (L + 2 * pad - NFFT) / HOP
        func samp(_ idx: Int) -> Float { idx < pad ? source[pad - idx] : (idx >= pad + L ? source[2 * L + pad - idx - 2] : source[idx - pad]) }
        var sStft = [Float](repeating: 0, count: 2 * FREQ * frames)
        for fi in 0..<frames { for k in 0..<FREQ { var re: Float = 0, im: Float = 0
            for n in 0..<NFFT { let v = samp(fi * HOP + n) * stftWin[n]; let a = 2 * Float.pi * Float(k) * Float(n) / Float(NFFT); re += v * cos(a); im += -v * sin(a) }
            sStft[k * frames + fi] = re; sStft[(FREQ + k) * frames + fi] = im } }
        // trunk graph
        let (hfn, hd) = hift
        var magOut = ndOut(hd, "magnitude", [1, FREQ, frames]); var phOut = ndOut(hd, "phase", [1, FREQ, frames])
        var hv = InferenceFunction.MutableViews(); hv.insert(&magOut, for: "magnitude"); hv.insert(&phOut, for: "phase")
        _ = try await hfn.run(inputs: ["mel": ndF32(hd, "mel", mel, [1, NMEL, T]), "s_stft": ndF32(hd, "s_stft", sStft, [1, 2 * FREQ, frames])],
                              states: InferenceFunction.MutableViews(), outputViews: consume hv)
        let mag = flattenAsFloat(magOut), phase = flattenAsFloat(phOut); let tf = mag.count / FREQ
        // iSTFT
        let outLen = (tf - 1) * HOP + NFFT - 2 * pad
        var wav = [Float](repeating: 0, count: outLen + NFFT), wsum = [Float](repeating: 0, count: outLen + NFFT)
        for fi in 0..<tf { var frame = [Float](repeating: 0, count: NFFT)
            for n in 0..<NFFT { var a2: Float = 0
                for k in 0..<FREQ { let re = mag[k * tf + fi] * cos(phase[k * tf + fi]), im = mag[k * tf + fi] * sin(phase[k * tf + fi])
                    let a = 2 * Float.pi * Float(k) * Float(n) / Float(NFFT); let w: Float = (k == 0 || k == NFFT / 2) ? 1 : 2; a2 += w * (re * cos(a) - im * sin(a)) }
                frame[n] = a2 / Float(NFFT) * stftWin[n] }
            let base = fi * HOP - pad
            for n in 0..<NFFT where base + n >= 0 && base + n < outLen { wav[base + n] += frame[n]; wsum[base + n] += stftWin[n] * stftWin[n] } }
        var out = [Float](repeating: 0, count: outLen)
        for i in 0..<outLen { out[i] = wsum[i] > 1e-8 ? wav[i] / wsum[i] : 0 }
        let realLen = min(out.count, realT * (out.count / VB))
        return realT == VB ? out : Array(out[0..<realLen])
    }

    // MARK: helpers

    private func ndF16(_ d: InferenceFunctionDescriptor, _ n: String, _ data: [Float], _ shape: [Int]) -> NDArray {
        var a = NDArray(descriptor: { guard case .ndArray(let nd)? = d.inputDescriptor(of: n) else { fatalError() }; return nd.resolvingDynamicDimensions(shape) }())
        fillNDArray(&a, as: Float16.self, with: data.map { Float16($0) }); return a
    }
    private func ndF32(_ d: InferenceFunctionDescriptor, _ n: String, _ data: [Float], _ shape: [Int]) -> NDArray {
        var a = NDArray(descriptor: { guard case .ndArray(let nd)? = d.inputDescriptor(of: n) else { fatalError() }; return nd.resolvingDynamicDimensions(shape) }())
        fillNDArray(&a, as: Float.self, with: data); return a
    }
    private func ndI32(_ d: InferenceFunctionDescriptor, _ n: String, _ data: [Int32], _ shape: [Int]) -> NDArray {
        var a = NDArray(descriptor: { guard case .ndArray(let nd)? = d.inputDescriptor(of: n) else { fatalError() }; return nd.resolvingDynamicDimensions(shape) }())
        fillNDArray(&a, as: Int32.self, with: data); return a
    }
    private func ndOut(_ d: InferenceFunctionDescriptor, _ n: String, _ shape: [Int]) -> NDArray {
        NDArray(descriptor: { guard case .ndArray(let nd)? = d.outputDescriptor(of: n) else { fatalError() }; return nd.resolvingDynamicDimensions(shape) }())
    }
    // T3 sampling: repetition_penalty 1.2, temperature 0.8, top_p 0.95.
    private static func sample(_ logits: [Float], gen: [Int], rng: inout UInt64) -> Int {
        let temp: Float = 0.8, repPen: Float = 1.2, topP: Float = 0.95
        var l = logits
        for t in Set(gen) where t < l.count { l[t] = l[t] > 0 ? l[t] / repPen : l[t] * repPen }
        let mx = l.max() ?? 0
        var probs = l.map { exp(($0 - mx) / temp) }; let sum = probs.reduce(0, +)
        for i in probs.indices { probs[i] /= max(1e-12, sum) }
        let order = probs.indices.sorted { probs[$0] > probs[$1] }
        var cum: Float = 0; var kept: [Int] = []
        for i in order { kept.append(i); cum += probs[i]; if cum >= topP { break } }
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        var r = Float((rng >> 33) & 0xFFFFFF) / Float(0x1000000) * cum
        for i in kept { r -= probs[i]; if r <= 0 { return i } }
        return kept.last ?? 0
    }
    private static func gaussian(_ n: Int, seed: UInt64) -> [Float] {
        var s = seed; func u() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float((s >> 33) & 0xFFFFFF) / Float(0x1000000) }
        var out = [Float](repeating: 0, count: n); var i = 0
        while i < n { let u1 = max(1e-7, u()), u2 = u(); let r = (-2 * log(u1)).squareRoot()
            out[i] = r * cos(2 * .pi * u2); if i + 1 < n { out[i + 1] = r * sin(2 * .pi * u2) }; i += 2 }
        return out
    }
    enum Err: Error { case load(String), desc, tooLong(Int) }
}

// Free loaders (init can't call self-scoped funcs before all stored properties are set).
private func loadGraph(_ dir: URL, _ name: String, _ opt: SpecializationOptions) async throws -> (InferenceFunction, InferenceFunctionDescriptor) {
    let m = try await AIModel(contentsOf: dir.appendingPathComponent(name).appendingPathComponent("\(name).aimodel"), options: opt)
    guard let fn = try m.loadFunction(named: "main"), let d = m.functionDescriptor(for: "main") else { throw ChatterboxTTS.Err.load(name) }
    return (fn, d)
}
private func readFloatBin(_ dir: URL, _ n: String) -> [Float] {
    guard let d = try? Data(contentsOf: dir.appendingPathComponent(n)) else { return [] }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

// NDArray fill/flatten helpers (mirrors CoreAIShared/NDArray+Helpers; inlined so coreai-audio
// needs no extra package dependency).
private func fillNDArray<T: BitwiseCopyable>(_ array: inout NDArray, as type: T.Type, with elements: some Collection<T>) {
    var view = array.mutableView(as: type)
    view.copyElements(fromContentsOf: elements)
}
private func flattenAsFloat(_ array: NDArray) -> [Float] {
    switch array.scalarType {
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    case .float16: return flattenNDArray(array, as: Float16.self)
    #endif
    case .float32: return flattenNDArray(array, as: Float.self)
    default: preconditionFailure("flattenAsFloat: unsupported scalar type \(array.scalarType)")
    }
}
private func flattenNDArray<T: BinaryFloatingPoint & BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [Float] {
    let shape = array.shape, rank = shape.count, total = shape.reduce(1, *)
    var result = [Float](repeating: 0, count: total)
    array.view(as: type).withUnsafePointer { ptr, s, strides in
        var expected = 1, contiguous = true
        for d in (0..<rank).reversed() { if strides[d] != expected { contiguous = false; break }; expected *= s[d] }
        if contiguous { for i in 0..<total { result[i] = Float(ptr[i]) }; return }
        var idx = [Int](repeating: 0, count: rank)
        for i in 0..<total {
            var off = 0; for d in 0..<rank { off += idx[d] * strides[d] }
            result[i] = Float(ptr[off])
            var dim = rank - 1
            while dim >= 0 { idx[dim] += 1; if idx[dim] < s[dim] { break }; idx[dim] = 0; dim -= 1 }
        }
    }
    return result
}
