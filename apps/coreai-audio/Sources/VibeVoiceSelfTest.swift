// Headless self-test for VibeVoice-Realtime-0.5B (VIBEVOICE_SELFTEST=1): load the 5 ported Core AI
// graphs (main LM 4L + tts LM 20L, both KV-stateful; diffusion head; connector; acoustic decoder),
// run the upstream streaming generate() loop with a host DPMSolver++ sampler, and gate the produced
// 24 kHz audio vs golden.f32 (the Python non-streaming oracle). Runs without the GUI (init()-launched).
//
// Mirrors conversion/vibevoice/host_e2e.py + ondevice/VibeVoiceRunner (Mac-validated cos 0.999).
// Uses the raw CoreAI runtime directly (GraphModel is stateless-only; the LMs need MutableViews KV
// state), so the two NDArray helpers are inlined here (no CoreAIShared dependency).
import CoreAI
import Foundation

// ---- inlined NDArray helpers (BSD-3, apple/coreai-models CoreAIShared) ----
private func fillND<T: BitwiseCopyable>(_ a: inout NDArray, as t: T.Type, with e: some Collection<T>) {
    var v = a.mutableView(as: t); v.copyElements(fromContentsOf: e)
}
private func flattenF16(_ a: NDArray) -> [Float] {
    a.view(as: Float16.self).withUnsafePointer { ptr, shape, _ in
        var n = 1; for i in 0..<shape.count { n *= shape[i] }   // Span<Int> has no reduce (non-escapable)
        var r = [Float](repeating: 0, count: n)
        for i in 0..<n { r[i] = Float(ptr[i]) }; return r
    }
}

enum VibeVoiceAssets {
    static let repo = "mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI"
    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("VibeVoiceAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VibeVoiceAssets")
        #endif
    }
    static var root: URL? {
        let p = location
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("meta.json").path) ? p : nil
    }
}

private struct SchedStep: Codable { let t: Int; let alpha: Double; let sigma: Double; let lambda: Double }
private struct VVMeta: Codable {
    let hidden, vae_dim, hop, n_text, main_prefill_len, tts_prefill_len, neg_prefill_len: Int
    let text_window, speech_window, num_noise: Int
    let cfg, scaling, bias: Double; let ddpm_steps: Int
    let schedule: [SchedStep]
    let main_layers, tts_layers, n_kv, head_dim: Int
}

private struct VVGraph {
    let d: InferenceFunctionDescriptor; let fn: InferenceFunction
    init(_ url: URL, _ opts: SpecializationOptions) async throws {
        let m = try await AIModel(contentsOf: url, options: opts)
        guard let d = m.functionDescriptor(for: "main"), let fn = try m.loadFunction(named: "main")
        else { throw NSError(domain: "vv", code: 1, userInfo: [NSLocalizedDescriptionKey: "load \(url.lastPathComponent)"]) }
        self.d = d; self.fn = fn
    }
}

func runVibeVoiceSelfTest() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let logURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["VV_RESULT"] ?? "/tmp/vv_result.txt")
    try? "".write(to: logURL, atomically: true, encoding: .utf8)
    func log(_ s: String) {
        print("[VV] \(s)")
        if let h = try? FileHandle(forWritingTo: logURL) { h.seekToEndOfFile(); h.write(Data(("[VV] \(s)\n").utf8)); try? h.close() }
    }
    func finish(_ c: Int32) -> Never { log("EXIT \(c)"); exit(c) }
    guard let root = VibeVoiceAssets.root else { log("FAIL: assets not at \(VibeVoiceAssets.location.path)"); finish(2) }

    func readF16(_ n: String) -> [Float] {
        let d = try! Data(contentsOf: root.appendingPathComponent(n))
        return d.withUnsafeBytes { $0.bindMemory(to: Float16.self).map { Float($0) } }
    }
    func readF32(_ n: String) -> [Float] {
        let d = try! Data(contentsOf: root.appendingPathComponent(n))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
    }
    let cu = ProcessInfo.processInfo.environment["VV_CU"] ?? "gpu"
    let kind: ComputeUnitKind = cu == "cpu" ? .cpu : cu == "ane" ? .neuralEngine : .gpu
    // Every graph here is fixed-shape (q=1 decode, fixed-T decoder), so do NOT ask for the reshape
    // hint: on iOS it makes the runtime skip the AOT specialization and compile on device, which
    // segfaults inside the MPSGraph AICode compiler. VV_EFR=1 restores the old behaviour.
    let efr = (ProcessInfo.processInfo.environment["VV_EFR"] ?? "0") != "0"
    var opts = SpecializationOptions(preferredComputeUnitKind: kind); opts.expectFrequentReshapes = efr

    do {
        let meta = try JSONDecoder().decode(VVMeta.self, from: Data(contentsOf: root.appendingPathComponent("meta.json")))
        let H = meta.hidden, VD = meta.vae_dim, DEC_T = 64
        func gurl(_ base: String) -> URL {
            log("load \(base)")                      // last line before a crash = the graph that failed
            for ext in ["h18p.aimodelc", "aimodelc", "aimodel"] {
                let u = root.appendingPathComponent("\(base).\(ext)"); if FileManager.default.fileExists(atPath: u.path) { return u }
                let u2 = root.appendingPathComponent("\(base)/\(base).\(ext)"); if FileManager.default.fileExists(atPath: u2.path) { return u2 }
            }
            log("FAIL: graph \(base) missing"); finish(2)
        }
        let t0 = Date()
        let mainG = try await VVGraph(gurl("vibevoice_mainlm_fp16_decode_cl512"), opts)
        let ttsG = try await VVGraph(gurl("vibevoice_ttslm_fp16_decode_cl512"), opts)
        let negG = try await VVGraph(gurl("vibevoice_ttslm_fp16_decode_cl512"), opts)
        let headG = try await VVGraph(gurl("vibevoice_diffusion_head_fp16"), opts)
        let connG = try await VVGraph(gurl("vibevoice_connector_fp16"), opts)
        let decG = try await VVGraph(gurl("vibevoice_decoder_fp16_t64"), opts)
        log(String(format: "loaded 6 graphs in %.1fs (%@)", Date().timeIntervalSince(t0), cu))

        func alloc(_ d: InferenceFunctionDescriptor, _ n: String, _ s: [Int], _ k: String) -> NDArray {
            let io = k == "input" ? d.inputDescriptor(of: n) : k == "state" ? d.stateDescriptor(of: n) : d.outputDescriptor(of: n)
            guard case .ndArray(let nd)? = io else { fatalError("\(n)") }
            return NDArray(descriptor: nd.resolvingDynamicDimensions(s))
        }
        func f16(_ a: inout NDArray, _ v: [Float]) { fillND(&a, as: Float16.self, with: v.map { Float16($0) }) }

        // ---- stateful KV backbone ----
        final class BB {
            let g: VVGraph, nl: Int, ctx: Int, H: Int; var kc: NDArray, vc: NDArray, pos = 0
            let kN: String, vN: String
            let allocF: (InferenceFunctionDescriptor, String, [Int], String) -> NDArray
            let f16F: (inout NDArray, [Float]) -> Void
            init(_ g: VVGraph, _ nl: Int, _ ctx: Int, _ H: Int,
                 _ allocF: @escaping (InferenceFunctionDescriptor, String, [Int], String) -> NDArray,
                 _ f16F: @escaping (inout NDArray, [Float]) -> Void) {
                self.g = g; self.nl = nl; self.ctx = ctx; self.H = H; self.allocF = allocF; self.f16F = f16F
                kN = g.d.stateNames[0]; vN = g.d.stateNames[1]
                kc = allocF(g.d, kN, [nl, 1, 2, ctx, 64], "state"); vc = allocF(g.d, vN, [nl, 1, 2, ctx, 64], "state")
            }
            func seed(_ K: [Float], _ V: [Float], _ L: Int) {
                var kb = [Float](repeating: 0, count: nl * 2 * ctx * 64), vb = kb
                for li in 0..<nl { for kv in 0..<2 { for p in 0..<L { for h in 0..<64 {
                    let s = ((li * 2 + kv) * L + p) * 64 + h, d = ((li * 2 + kv) * ctx + p) * 64 + h
                    kb[d] = K[s]; vb[d] = V[s]
                }}}}
                f16F(&kc, kb); f16F(&vc, vb); pos = L
            }
            func step(_ emb: [Float]) async throws -> [Float] {
                var ie = allocF(g.d, "inputs_embeds", [1, 1, H], "input"); f16F(&ie, emb)
                var pa = allocF(g.d, "pos", [1], "input"); fillND(&pa, as: Int32.self, with: [Int32(pos)])
                var ho = allocF(g.d, "hidden", [1, 1, H], "output")
                var st = InferenceFunction.MutableViews(); st.insert(&kc, for: kN); st.insert(&vc, for: vN)
                var ov = InferenceFunction.MutableViews(); ov.insert(&ho, for: "hidden")
                _ = try await g.fn.run(inputs: ["inputs_embeds": ie, "pos": pa], states: consume st, outputViews: consume ov)
                pos += 1; return flattenF16(ho)
            }
        }
        let mainBB = BB(mainG, meta.main_layers, 512, H, alloc, f16)
        let ttsBB = BB(ttsG, meta.tts_layers, 512, H, alloc, f16)
        let negBB = BB(negG, meta.tts_layers, 512, H, alloc, f16)
        mainBB.seed(readF16("main_k.f16"), readF16("main_v.f16"), meta.main_prefill_len)
        ttsBB.seed(readF16("tts_k.f16"), readF16("tts_v.f16"), meta.tts_prefill_len)
        negBB.seed(readF16("neg_k.f16"), readF16("neg_v.f16"), meta.neg_prefill_len)

        let textEmbeds = readF16("text_embeds.f16"), typeEmb = readF16("type_emb.f16")
        let negLast = readF16("negtts_last.f16"), noise = readF16("noise.f16")
        let eW1 = readF32("eos_fc1_w.f32"), eB1 = readF32("eos_fc1_b.f32"), eW2 = readF32("eos_fc2_w.f32"), eB2 = readF32("eos_fc2_b.f32")
        func typeVec(_ i: Int) -> [Float] { Array(typeEmb[i * H..<(i + 1) * H]) }
        func addv(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map(+) }

        func headEps(_ noisy2: [Float], _ t2: [Float], _ cond2: [Float]) async throws -> [Float] {
            var n = alloc(headG.d, "noisy_images", [2, VD], "input"); f16(&n, noisy2)
            var t = alloc(headG.d, "timesteps", [2], "input"); f16(&t, t2)
            var c = alloc(headG.d, "condition", [2, H], "input"); f16(&c, cond2)
            var o = alloc(headG.d, "eps", [2, VD], "output"); var ov = InferenceFunction.MutableViews(); ov.insert(&o, for: "eps")
            _ = try await headG.fn.run(inputs: ["noisy_images": n, "timesteps": t, "condition": c], states: InferenceFunction.MutableViews(), outputViews: consume ov)
            return flattenF16(o)
        }
        func connEmbed(_ lat: [Float]) async throws -> [Float] {
            var f = alloc(connG.d, "features", [1, 1, VD], "input"); f16(&f, lat)
            var o = alloc(connG.d, "embed", [1, 1, H], "output"); var ov = InferenceFunction.MutableViews(); ov.insert(&o, for: "embed")
            _ = try await connG.fn.run(inputs: ["features": f], states: InferenceFunction.MutableViews(), outputViews: consume ov)
            return flattenF16(o)
        }
        func decodeAudio(_ latsT: [Float]) async throws -> [Float] {
            var l = alloc(decG.d, "latents", [1, VD, DEC_T], "input"); f16(&l, latsT)
            var o = alloc(decG.d, "audio", [1, 1, DEC_T * meta.hop], "output"); var ov = InferenceFunction.MutableViews(); ov.insert(&o, for: "audio")
            _ = try await decG.fn.run(inputs: ["latents": l], states: InferenceFunction.MutableViews(), outputViews: consume ov)
            return flattenF16(o)
        }
        func eos(_ h: [Float]) -> Float {
            var x = [Float](repeating: 0, count: H)
            for i in 0..<H { var a = eB1[i]; for j in 0..<H { a += eW1[i * H + j] * h[j] }; x[i] = max(0, a) }
            var a = eB2[0]; for j in 0..<H { a += eW2[j] * x[j] }; return 1 / (1 + exp(-a))
        }
        func ddpm(_ posC: [Float], _ negC: [Float], _ ni: Int) async throws -> [Float] {
            let S = meta.schedule
            var x = Array(noise[ni * 2 * VD ..< ni * 2 * VD + VD]).map { Double($0) }
            let cond2 = posC + negC; var mPrev: [Double]? = nil
            for i in 0..<S.count {
                let xf = x.map { Float($0) }
                let e = try await headEps(xf + xf, [Float(S[i].t), Float(S[i].t)], cond2)
                var v = [Double](repeating: 0, count: VD)
                for j in 0..<VD { let ce = Double(e[j]), ue = Double(e[VD + j]); v[j] = ue + meta.cfg * (ce - ue) }
                let a = S[i].alpha, s = S[i].sigma, lam = S[i].lambda
                let aN = i + 1 < S.count ? S[i + 1].alpha : 1.0, sN = i + 1 < S.count ? S[i + 1].sigma : 0.0
                let lamN = i + 1 < S.count ? S[i + 1].lambda : 20.0
                let h = lamN - lam, em1 = exp(-h) - 1.0
                var m0 = [Double](repeating: 0, count: VD); for j in 0..<VD { m0[j] = a * x[j] - s * v[j] }
                if i > 0 && i < S.count - 1, let mp = mPrev {
                    let r0 = (lam - S[i - 1].lambda) / h
                    for j in 0..<VD { let D1 = (m0[j] - mp[j]) / r0; x[j] = (sN / s) * x[j] - aN * em1 * (m0[j] + 0.5 * D1) }
                } else { for j in 0..<VD { x[j] = (sN / s) * x[j] - aN * em1 * m0[j] } }
                mPrev = m0
            }
            return x.map { Float($0) }
        }

        // ---- generate loop ----
        let TW = meta.text_window, SW = meta.speech_window
        var latents: [[Float]] = [], negCond = negLast, ttsLast = [Float](repeating: 0, count: H)
        var winIdx = 0, noiseI = 0, finished = false
        let g0 = Date()
        while !finished {
            let lo = winIdx * TW, hi = min(lo + TW, meta.n_text); winIdx += 1
            if lo < hi {
                for k in lo..<hi {
                    let mh = try await mainBB.step(Array(textEmbeds[k * H..<(k + 1) * H]))
                    ttsLast = try await ttsBB.step(addv(mh, typeVec(1)))
                }
            }
            for _ in 0..<SW {
                let lat = try await ddpm(ttsLast, negCond, min(noiseI, meta.num_noise - 1)); noiseI += 1
                latents.append(lat)
                let emb = try await connEmbed(lat)
                ttsLast = try await ttsBB.step(addv(emb, typeVec(0)))
                negCond = try await negBB.step(addv(emb, typeVec(0)))
                if eos(ttsLast) > 0.5 || latents.count >= DEC_T { finished = true; break }
            }
            if lo >= hi && winIdx * TW > meta.n_text + TW { finished = true }
        }
        let genSec = Date().timeIntervalSince(g0), N = latents.count
        log(String(format: "%d latents / %.2fs audio in %.1fs (%.1f tok/s)", N, Double(N * meta.hop) / 24000, genSec, Double(N) / genSec))

        var latsT = [Float](repeating: 0, count: VD * DEC_T)
        for n in 0..<N { for c in 0..<VD { latsT[c * DEC_T + n] = latents[n][c] / Float(meta.scaling) - Float(meta.bias) } }
        let audio = Array((try await decodeAudio(latsT))[0..<(N * meta.hop)])

        // gate vs golden.f32 (Python non-streaming oracle), first N frames
        if let gold = try? Data(contentsOf: root.appendingPathComponent("golden.f32")).withUnsafeBytes({ Array($0.bindMemory(to: Float.self)) }) {
            let m = min(audio.count, gold.count)
            var dot = 0.0, na = 0.0, nb = 0.0
            for k in 0..<m { let a = Double(audio[k]), b = Double(gold[k]); dot += a * b; na += a * a; nb += b * b }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
            log(String(format: "gate vs golden: cos=%.6f  rms=%.4f  -> %@", cos,
                       (audio.reduce(0) { $0 + $1 * $1 } / Float(audio.count)).squareRoot(), cos >= 0.99 ? "PASS" : "FAIL"))
        } else {
            log(String(format: "no golden.f32; produced %.2fs, rms=%.4f", Double(audio.count) / 24000,
                       (audio.reduce(0) { $0 + $1 * $1 } / Float(audio.count)).squareRoot()))
        }
        finish(0)
    } catch { log("FAIL: \(error)"); finish(1) }
}
