// Z-Image-Turbo (Tongyi-MAI, 6B, Apache-2.0) — bespoke host loop.
//
// The high-level CoreAIDiffusionPipeline can't drive this one: the DiT graph consumes
// host-prepped tokens + RoPE, and Z-Image's CFG is negated. Everything the host needs
// beyond the three big graphs ships as ~2.4 MB of glue, so nothing is re-implemented
// from the reference by hand:
//
//   zimage_encoder.aimodel     input_ids [1,64] + additive mask -> penultimate hidden
//                              (embed_tokens is INSIDE the graph — no 778 MB matrix here)
//   zimage_t_embedder.aimodel  timestep [1] -> adaln [1,256]   (no timestep MLP port)
//   rope_axis{0,1,2}.f32       RopeEmbedder is `cat(freqs[i][ids[:,i]])` — a per-axis
//                              lookup, so three tables reproduce it exactly for any
//                              resolution and prompt length
//   zimage_dit.aimodel         bf16, one graph for 256/512/1024 and any prompt length
//   zimage_vae_<side>.aimodel  fp32 (fp16 overflows the VAE to NaN)
//
// Three details each cost a wrong image (all verified against the fp32 diffusers run):
//   1. the DiT conditions on the encoder's PENULTIMATE hidden state
//   2. the CFG is NEGATED: noise = -(pos + g*(pos - neg))
//   3. the caption is padded to a multiple of 32 with a *learned* pad token that is real
//      attention context, so n_cap = roundUp(L,32) and cond/uncond usually differ

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers

// The DiT and the encoder keep bf16 weights and bf16 compute, but their graphs are
// exported with fp32 boundaries (`export_dit.py --io-fp32`). That is not a preference:
// Swift cannot fill or read a bfloat16 NDArray — CoreAIRuntime.BFloat16 is not public,
// and a UInt16 view trips the runtime's element-type check.

final class ZImagePipeline {
    // Graph + glue names inside the bundle directory.
    private enum F {
        static let dit = "zimage_dit"
        static let encoder = "zimage_encoder"
        static let tEmbedder = "zimage_t_embedder"
        static let vaePrefix = "zimage_vae_"
    }

    private let dir: URL
    private let encLen = 64          // the encoder graph is a fixed 64 chat-templated tokens
    private let seqMultiple = 32     // SEQ_MULTI_OF
    private let latentChannels = 16
    private let ropeFreqs = 64       // head_dim / 2

    private var dit: (InferenceFunction, InferenceFunctionDescriptor)?
    private var enc: (InferenceFunction, InferenceFunctionDescriptor)?
    private var tEmb: (InferenceFunction, InferenceFunctionDescriptor)?
    private var vae: [Int: (InferenceFunction, InferenceFunctionDescriptor)] = [:]
    private var tokenizer: Tokenizer?
    /// rope[axis] = rows × freqs × (cos, sin)
    private var rope: [[[SIMD2<Float>]]] = []

    init(dir: URL) { self.dir = dir }

    static func looksLikeZImage(_ url: URL) -> Bool { resolve(in: url, contains: F.dit) != nil }

    // MARK: - load

    func load() async throws {
        guard let ditURL = Self.resolve(in: dir, contains: F.dit),
              let encURL = Self.resolve(in: dir, contains: F.encoder),
              let temURL = Self.resolve(in: dir, contains: F.tEmbedder) else {
            throw Self.err("bundle is missing a Z-Image graph (dit / encoder / t_embedder)")
        }
        dit = try await Self.loadModel(ditURL)
        enc = try await Self.loadModel(encURL)
        tEmb = try await Self.loadModel(temURL)
        for side in [256, 512, 1024] {
            if let u = Self.resolve(in: dir, contains: "\(F.vaePrefix)\(side)") {
                vae[side] = try await Self.loadModel(u)
            }
        }
        guard !vae.isEmpty else { throw Self.err("no zimage_vae_<side>.aimodel in the bundle") }

        tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"))
        rope = try Self.loadRopeTables(dir)
    }

    var availableSides: [Int] { vae.keys.sorted() }

    // MARK: - generate

    /// `initialLatent` exists so the CLI runner can replay the Python reference's noise and
    /// compare images pixel-for-pixel; the app always passes nil and seeds its own.
    func generate(prompt: String, negativePrompt: String, side: Int, steps: Int, guidance: Float,
                  seed: UInt64, initialLatent: [Float]? = nil,
                  progress: @escaping @Sendable (Int, Int) -> Bool) async throws -> CGImage {
        guard let (ditFn, ditD) = dit, let (vaeFn, vaeD) = vae[side] else {
            throw Self.err("Z-Image not loaded for side \(side)")
        }
        let lat = side / 8, grid = lat / 2, nImg = grid * grid

        let cond = try await encode(prompt)
        let uncond = try await encode(negativePrompt)

        var latent = initialLatent ?? Self.noise(count: latentChannels * lat * lat, seed: seed)

        for s in 0..<steps {
            if !progress(s, steps) { throw CancellationError() }
            let sigma = Self.sigma(s, steps), sigmaNext = Self.sigma(s + 1, steps)
            let adaln = try await timestepEmbed(1.0 - sigma)

            let img = Self.patchify(latent, lat: lat, grid: grid)
            let pos = try await velocity(ditFn, ditD, img: img, nImg: nImg, cap: cond, adaln: adaln)
            var pred = pos
            if guidance > 0 {
                let neg = try await velocity(ditFn, ditD, img: img, nImg: nImg, cap: uncond, adaln: adaln)
                for i in 0..<pred.count { pred[i] = pos[i] + guidance * (pos[i] - neg[i]) }
            }
            // Z-Image's CFG is negated, then a FlowMatchEuler step.
            let dSigma = sigmaNext - sigma
            let vel = Self.unpatchify(pred, lat: lat, grid: grid)
            for i in 0..<latent.count { latent[i] += dSigma * -vel[i] }
        }
        _ = progress(steps, steps)

        // The VAE graph bakes the un-scale (z/0.3611 + 0.1159) in itself — feed the RAW latent.
        var z = alloc(vaeD, "z", [1, latentChannels, lat, lat], .input)
        fillF(&z, latent)
        var out = alloc(vaeD, "image", [1, 3, side, side], .output)
        var views = InferenceFunction.MutableViews(); views.insert(&out, for: "image")
        _ = try await vaeFn.run(inputs: ["z": z], outputViews: consume views)

        guard let cg = Self.makeCGImage(rgb: flattenAsFloat(out), side: side) else {
            throw Self.err("VAE → CGImage failed")
        }
        return cg
    }

    // MARK: - stages

    /// prompt -> caption embeds (the encoder's PENULTIMATE hidden), valid rows only.
    private func encode(_ text: String) async throws -> [[Float]] {
        guard let (fn, d) = enc, let tok = tokenizer else { throw Self.err("encoder not loaded") }
        // The reference tokenizes with add_generation_prompt=True; the convenience overload's
        // default is not something to assume.
        let templated = try tok.applyChatTemplate(
            messages: [["role": "user", "content": text]],
            chatTemplate: nil, addGenerationPrompt: true, truncation: false,
            maxLength: nil, tools: nil)
        var ids = templated
        FileHandle.standardError.write("[zimage] chat-template tokens: \(ids.count)\n".data(using: .utf8)!)
        if ids.count > encLen {
            throw Self.err("prompt is \(ids.count) tokens; this encoder graph is fixed at \(encLen)")
        }
        let valid = ids.count
        let padId = tok.unknownTokenId ?? 0
        ids += Array(repeating: padId, count: encLen - valid)

        var idArr = alloc(d, "input_ids", [1, encLen], .input)
        fillNDArray(&idArr, as: Int32.self, with: ids.map(Int32.init))

        // additive mask: causal AND non-padding (pads are keys nobody may attend to)
        let neg = -Float.greatestFiniteMagnitude / 2
        var maskArr = alloc(d, "mask", [1, 1, encLen, encLen], .input)
        var mask = [Float](repeating: 0, count: encLen * encLen)
        for q in 0..<encLen {
            for k in 0..<encLen where k > q || k >= valid { mask[q * encLen + k] = neg }
        }
        fillF(&maskArr, mask)

        var out = alloc(d, "penultimate", [1, encLen, 2560], .output)
        var views = InferenceFunction.MutableViews(); views.insert(&out, for: "penultimate")
        _ = try await fn.run(inputs: ["input_ids": idArr, "mask": maskArr], outputViews: consume views)

        let flat = flattenAsFloat(out)
        return (0..<valid).map { Array(flat[$0 * 2560..<($0 + 1) * 2560]) }
    }

    private func timestepEmbed(_ t: Float) async throws -> [Float] {
        guard let (fn, d) = tEmb else { throw Self.err("t_embedder not loaded") }
        var tArr = alloc(d, "timestep", [1], .input)
        fillF(&tArr, [t])
        var out = alloc(d, "adaln", [1, 256], .output)
        var views = InferenceFunction.MutableViews(); views.insert(&out, for: "adaln")
        _ = try await fn.run(inputs: ["timestep": tArr], outputViews: consume views)
        return flattenAsFloat(out)
    }

    /// One DiT forward: host-prepped tokens + RoPE in, patch-space velocity out.
    private func velocity(_ fn: InferenceFunction, _ d: InferenceFunctionDescriptor,
                          img: [Float], nImg: Int, cap: [[Float]], adaln: [Float]) async throws -> [Float] {
        let valid = cap.count
        let nCap = ((valid + seqMultiple - 1) / seqMultiple) * seqMultiple
        let grid = Int(Double(nImg).squareRoot().rounded())

        // caption: pad rows are substituted with the learned pad token in-graph, so their
        // contents don't matter — only the mask does.
        var capFlat = [Float](repeating: 0, count: nCap * 2560)
        for i in 0..<valid { capFlat.replaceSubrange(i * 2560..<(i + 1) * 2560, with: cap[i]) }
        var capPad = [Float](repeating: 0, count: nCap)
        for i in valid..<nCap { capPad[i] = 1 }

        // RoPE: caption token i -> (i+1, 0, 0); image token (h,w) -> (nCap+1, h, w)
        var capCos = [Float](), capSin = [Float]()
        capCos.reserveCapacity(nCap * ropeFreqs); capSin.reserveCapacity(nCap * ropeFreqs)
        for i in 0..<nCap { appendRope(&capCos, &capSin, i + 1, 0, 0) }
        var xCos = [Float](), xSin = [Float]()
        xCos.reserveCapacity(nImg * ropeFreqs); xSin.reserveCapacity(nImg * ropeFreqs)
        for h in 0..<grid { for w in 0..<grid { appendRope(&xCos, &xSin, nCap + 1, h, w) } }

        var inputs: [String: NDArray] = [:]
        func put(_ name: String, _ shape: [Int], _ v: [Float]) {
            var a = alloc(d, name, shape, .input); fillF(&a, v); inputs[name] = a
        }
        put("img_tokens", [1, nImg, 64], img)
        put("cap_feats", [1, nCap, 2560], capFlat)
        put("adaln", [1, 256], adaln)
        put("x_cos", [1, nImg, ropeFreqs], xCos)
        put("x_sin", [1, nImg, ropeFreqs], xSin)
        put("cap_cos", [1, nCap, ropeFreqs], capCos)
        put("cap_sin", [1, nCap, ropeFreqs], capSin)
        put("x_pad_mask", [1, nImg, 1], [Float](repeating: 0, count: nImg))
        put("cap_pad_mask", [1, nCap, 1], capPad)

        var out = alloc(d, "velocity", [1, nImg + nCap, 64], .output)
        var views = InferenceFunction.MutableViews(); views.insert(&out, for: "velocity")
        _ = try await fn.run(inputs: inputs, outputViews: consume views)
        return Array(flattenAsFloat(out)[0..<(nImg * 64)])   // image slice only
    }

    private func appendRope(_ cos: inout [Float], _ sin: inout [Float], _ t: Int, _ h: Int, _ w: Int) {
        for (axis, idx) in [(0, t), (1, h), (2, w)] {
            for v in rope[axis][idx] { cos.append(v.x); sin.append(v.y) }
        }
    }

    // MARK: - host math

    /// σ_i = s(1-t)/(s(1-t)+t), t = i/(n-1), s = 3 — matches the diffusers scheduler to 3e-8
    /// for every step count and resolution (the mu shift is not applied by this pipeline).
    static func sigma(_ i: Int, _ n: Int) -> Float {
        if i >= n { return 0 }
        let t = Float(i) / Float(max(n - 1, 1)), s: Float = 3
        return s * (1 - t) / (s * (1 - t) + t)
    }

    /// latent [C,H,W] -> tokens [grid*grid, 64], feature f = (dy*2+dx)*C + c  (verified exact)
    static func patchify(_ latent: [Float], lat: Int, grid: Int) -> [Float] {
        let C = 16
        var out = [Float](repeating: 0, count: grid * grid * 64)
        for h in 0..<grid {
            for w in 0..<grid {
                let k = h * grid + w
                for dy in 0..<2 {
                    for dx in 0..<2 {
                        let base = (dy * 2 + dx) * C
                        let y = 2 * h + dy, x = 2 * w + dx
                        for c in 0..<C { out[k * 64 + base + c] = latent[c * lat * lat + y * lat + x] }
                    }
                }
            }
        }
        return out
    }

    static func unpatchify(_ tokens: [Float], lat: Int, grid: Int) -> [Float] {
        let C = 16
        var out = [Float](repeating: 0, count: C * lat * lat)
        for h in 0..<grid {
            for w in 0..<grid {
                let k = h * grid + w
                for dy in 0..<2 {
                    for dx in 0..<2 {
                        let base = (dy * 2 + dx) * C
                        let y = 2 * h + dy, x = 2 * w + dx
                        for c in 0..<C { out[c * lat * lat + y * lat + x] = tokens[k * 64 + base + c] }
                    }
                }
            }
        }
        return out
    }

    /// Deterministic standard normal (Box–Muller over a splitmix64 stream).
    static func noise(count: Int, seed: UInt64) -> [Float] {
        var s = seed &+ 0x9E3779B97F4A7C15
        func next() -> Double {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= (z >> 31)
            return Double(z >> 11) * (1.0 / 9007199254740992.0)
        }
        var out = [Float](); out.reserveCapacity(count)
        while out.count < count {
            let u1 = max(next(), 1e-12), u2 = next()
            let r = (-2 * Foundation.log(u1)).squareRoot(), th = 2 * Double.pi * u2
            out.append(Float(r * Foundation.cos(th)))
            if out.count < count { out.append(Float(r * Foundation.sin(th))) }
        }
        return out
    }

    // MARK: - plumbing

    private enum IOKind { case input, output }

    private func descriptor(_ d: InferenceFunctionDescriptor, _ name: String, _ k: IOKind) -> NDArrayDescriptor {
        let io = k == .input ? d.inputDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) is not an ndArray") }
        return nd
    }

    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ k: IOKind) -> NDArray {
        NDArray(descriptor: descriptor(d, name, k).resolvingDynamicDimensions(shape))
    }

    private func fillF(_ a: inout NDArray, _ v: [Float]) {
        switch a.scalarType {
        case .float32: fillNDArray(&a, as: Float.self, with: v)
        case .float16: fillNDArray(&a, as: Float16.self, with: v.map { Float16($0) })
        default: fatalError("fillF on \(a.scalarType)")
        }
    }

    private static func loadModel(_ url: URL) async throws -> (InferenceFunction, InferenceFunctionDescriptor) {
        // AOT bundles load with .default; AIModel(contentsOf:) does not follow symlinks.
        let m = try await AIModel(contentsOf: url.resolvingSymlinksInPath(), options: .default)
        guard let d = m.functionDescriptor(for: "main"), let f = try m.loadFunction(named: "main") else {
            throw err("no 'main' function in \(url.lastPathComponent)")
        }
        return (f, d)
    }

    private static func resolve(in dir: URL, contains needle: String) -> URL? {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: dir.path) else { return nil }
        // prefer the AOT .aimodelc when both are present
        for ext in ["aimodelc", "aimodel"] {
            if let hit = items.first(where: { $0.contains(needle) && $0.hasSuffix(ext) }) {
                return dir.appendingPathComponent(hit)
            }
        }
        return nil
    }

    private static func loadRopeTables(_ dir: URL) throws -> [[[SIMD2<Float>]]] {
        struct Axis: Decodable { let rows: Int; let freqs: Int }
        struct Meta: Decodable { let axes: [Axis] }
        let meta = try JSONDecoder().decode(
            Meta.self, from: Data(contentsOf: dir.appendingPathComponent("rope_meta.json")))
        return try meta.axes.enumerated().map { i, ax in
            let d = try Data(contentsOf: dir.appendingPathComponent("rope_axis\(i).f32"))
            let f = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return (0..<ax.rows).map { r in
                (0..<ax.freqs).map { c in
                    let o = (r * ax.freqs + c) * 2
                    return SIMD2<Float>(f[o], f[o + 1])
                }
            }
        }
    }

    private static func makeCGImage(rgb: [Float], side: Int) -> CGImage? {
        var px = [UInt8](repeating: 255, count: side * side * 4)
        let plane = side * side
        for i in 0..<plane {
            for c in 0..<3 {
                let v = (min(max(rgb[c * plane + i], -1), 1) + 1) * 127.5
                px[i * 4 + c] = UInt8(v.rounded())
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &px, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: side * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        return ctx.makeImage()
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "ZImagePipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
