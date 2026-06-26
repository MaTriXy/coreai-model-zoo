// Gemma4VLBackend — Gemma 4 E2B vision on the pipelined engine: the zoo's
// second VLM, riding the SAME QAT checkpoint + PLE tables as the text modes.
//
// Three artifact sets under Documents/models/:
//   * gemma4_e2b_qat_vl_decode_int4linsym_aotc — the text decoder (PROVIDER
//     mode + AOT h18p: the tbl in-graph table gather overflows the iOS
//     MPSGraph ~208 KB per-encode scratch heap — an engine bug; provider
//     keeps the image splice in-graph and feeds PLE rows per token).
//   * gemma4_e2b_qat_vl_vision — the fixed-grid ViT .aimodel, run ONCE per
//     image: patches [2304, 768] f16 -> image_embeds [256, 1536].
//   * gemma4_qat_gather_raw — the PLE int8 dump (shared with the text modes;
//     mmap, rows gathered per token by the PerTokenInputProvider).
//
// Host contract (mirrors the gated python pipeline + the device probe):
//   * preprocess: resize 768x768, /255 to [0,1] (the graph scales 2x-1
//     in-graph), ROW-MAJOR patchify, per-patch [py, px, c] (HWC) flatten.
//   * prompt: <bos><|turn>user\n<|image> + 256x<|image|> + <image|> + text
//     + <turn|>\n<|turn>model\n — soft-token ids REWRITTEN to extension ids
//     V+slot; the engine binds image_embeds [280,1536] as a static buffer
//     (square images fill rows 0..255) alongside the per-token provider.
//   * PLE: extension ids (>= V) gather the PAD row (id 0) — HF's
//     llm_input_ids[mm_mask] = pad rule, applied host-side in the provider.
//
// Numerics: Mac engine provider-mode 24/24 vs the python gate; device prefix
// + margin rule + image-aware 64-token rollout (GEMMA4VL_STATE.md).

import CoreAI
import CoreAILanguageModels
import CoreAIShared
import CoreGraphics
import Foundation
import Metal
import Tokenizers

@MainActor
final class Gemma4VLBackend {
    static let decoderBundle = "gemma4_e2b_qat_vl_decode_int4linsym_aotc"
    static let visionDir = "gemma4_e2b_qat_vl_vision"
    static let tablesDir = "gemma4_qat_gather_raw"
    static let hfDecoderPath = "gpu-pipelined/gemma4_e2b_qat_vl_decode_int4linsym_aotc_h18p"
    static let hfVisionPath = "gpu-pipelined/gemma4_e2b_qat_vl_vision"
    static let hfTablesPath = "ios-frontend/gemma4_qat_gather_raw"
    static let label = "Gemma 4 VL ⚡pipelined"

    // Architecture constants (768x768 square grid)
    private let V: Int32 = 262_144
    private let N_SLOTS = 280    // image_embeds static-input rows (max budget)
    private let N_SOFT = 256     // soft tokens for the square grid (48*48/9)
    private let HID = 1536
    private let PATCHES = 2304   // 48x48
    private let PATCH_DIM = 768  // 16*16*3 (HWC)
    private let EOT: Int32 = 106 // <turn|> (gemma ends turns here; eos is <eos> 1)

    // Real vocab strings of the special ids (NOT the friendly aliases —
    // "<start_of_image>" etc. tokenize as plain text):
    //   105 <|turn>  106 <turn|>  255999 <|image>  258880 <|image|>  258882 <image|>
    private let SOFT_ID: Int32 = 258_880

    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    private var visionModel: AIModel?
    private var visionFn: InferenceFunction?
    private var visionD: InferenceFunctionDescriptor?
    private var table: PLERowTable?
    private(set) var ctx = 4096

    // Owned static-input buffer (alive for the engine's lifetime)
    private var imgBuf: MTLBuffer?

    private(set) var imageAttached = false

    var loaded: Bool { engine != nil }

    func load() async throws {
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let models = docs.appendingPathComponent("models")
        let bundle = try LanguageBundle(at: models.appendingPathComponent(Self.decoderBundle))
        ctx = bundle.maxContextLength

        guard let device = MTLCreateSystemDefaultDevice() else { throw Self.err("no Metal device") }
        let img = device.makeBuffer(length: N_SLOTS * HID * 2, options: .storageModeShared)!
        memset(img.contents(), 0, img.length)
        imgBuf = img

        // PLE rows per token from the int8 mmap dump; extension ids (image
        // slots) read the PAD row — the host half of the splice contract.
        let tbl = try PLERowTable(rawDir: models.appendingPathComponent(Self.tablesDir))
        table = tbl
        let vocab = V
        let provider: PerTokenInputProvider = { _, token, _, destination, byteCount in
            let t = token >= vocab ? Int32(0) : token
            tbl.fill(token: t, destination: destination, byteCount: byteCount)
        }

        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
            options: EngineOptions(
                perTokenInputProvider: provider,
                staticInputBuffers: ["image_embeds": StaticInputBuffer(img)])
        )
        tokenizer = try await bundle.loadTokenizer()

        // vision tower (plain .aimodel, GPU)
        let visURL = models.appendingPathComponent(Self.visionDir)
            .appendingPathComponent("\(Self.visionDir).aimodel")
        guard FileManager.default.fileExists(atPath: visURL.path) else {
            throw Self.err("missing \(Self.visionDir).aimodel in Documents/models/\(Self.visionDir)")
        }
        var vo = SpecializationOptions(preferredComputeUnitKind: .gpu)
        vo.expectFrequentReshapes = false
        let vm = try await AIModel(contentsOf: visURL, options: vo)
        guard let fn = try vm.loadFunction(named: "main") else { throw Self.err("vision main missing") }
        visionModel = vm
        visionFn = fn
        visionD = fn.descriptor

        // 1-token warmup through the decoder graph
        _ = try await run(ids: [2364], maxTokens: 1, eos: nil, onText: { _ in })
        print(PipelinedBackend.memLine("\(Self.label) loaded"))
    }

    func unload() {
        engine = nil
        tokenizer = nil
        visionFn = nil
        visionModel = nil
        table = nil
        imgBuf = nil
        imageAttached = false
    }

    // MARK: - Image attach

    /// Preprocess + vision-encode `cgImage` and write the 256 soft-token rows
    /// into the decoder's image_embeds buffer. Stays attached across turns
    /// until replaced (each generate re-prefills).
    func attach(cgImage: CGImage) async throws {
        guard let visionFn, let visionD, let imgBuf else {
            throw Self.err("backend not loaded")
        }
        let t0 = SuspendingClock.now
        let patches = Self.preprocess(cgImage: cgImage)

        guard case .ndArray(let pin)? = visionD.inputDescriptor(of: "patches") else {
            throw Self.err("patches input missing")
        }
        var pArr = NDArray(descriptor: pin.resolvingDynamicDimensions([PATCHES, PATCH_DIM]))
        fillNDArray(&pArr, as: Float16.self, with: patches)

        guard case .ndArray(let e0)? = visionD.outputDescriptor(of: "image_embeds") else {
            throw Self.err("vision output missing")
        }
        var embOut = NDArray(descriptor: e0.resolvingDynamicDimensions([N_SOFT, HID]))
        var out = InferenceFunction.MutableViews()
        out.insert(&embOut, for: "image_embeds")
        _ = try await visionFn.run(
            inputs: ["patches": pArr],
            states: InferenceFunction.MutableViews(),
            outputViews: consume out)

        let emb = flattenAsFloat(embOut)
        let imgP = imgBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<emb.count { imgP[i] = Float16(emb[i]) }
        imageAttached = true
        let dt = Self.seconds(since: t0)
        print(String(format: "[gemma4vl] image attached (vision %.0f ms)", dt * 1000))
    }

    func detachImage() {
        imageAttached = false
        if let imgBuf { memset(imgBuf.contents(), 0, imgBuf.length) }
    }

    // MARK: - Generation

    func generate(_ prompt: String, maxNew: Int,
                  onUpdate: (String) -> Void) async throws -> PipelinedBackend.GenStats {
        guard let tokenizer else { throw Self.err("backend not loaded") }
        var ids: [Int32]
        if imageAttached {
            // Gemma turn with the image block; encode then rewrite softs.
            let text = "<bos><|turn>user\n<|image>"
                + String(repeating: "<|image|>", count: N_SOFT)
                + "<image|>\(prompt)<turn|>\n<|turn>model\n"
            ids = tokenizer.encode(text: text).map { Int32($0) }
            var slot: Int32 = 0
            for i in 0..<ids.count where ids[i] == SOFT_ID {
                ids[i] = V + slot
                slot += 1
            }
            guard slot == Int32(N_SOFT) else {
                throw Self.err("expected \(N_SOFT) image soft tokens, found \(slot)")
            }
        } else {
            // Same fallback as the gemma text modes (explicit <bos>).
            let text = "<bos><|turn>user\n\(prompt)<turn|>\n<|turn>model\n"
            ids = tokenizer.encode(text: text).map { Int32($0) }
        }
        guard ids.count < ctx - 1 else {
            throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(ctx)")
        }
        let budget = min(maxNew, ctx - ids.count - 1)
        let stats = try await run(ids: ids, maxTokens: budget, eos: tokenizer.eosTokenId) { gen in
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
        print(PipelinedBackend.memLine("\(Self.label) gen"))
        return stats
    }

    private func run(
        ids: [Int32], maxTokens: Int, eos: Int?, onText: ([Int]) -> Void
    ) async throws -> PipelinedBackend.GenStats {
        guard let engine else { throw Self.err("backend not loaded") }
        try await engine.reset()
        let stream = try engine.generate(
            with: ids,
            samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens)
        )
        var stats = PipelinedBackend.GenStats(label: Self.label, prefillTok: ids.count)
        var gen: [Int] = []
        let t0 = SuspendingClock.now
        var tGen = t0
        // Throttle live UI to ~25 fps + exclude the UI-callback time from the decode
        // timer (the whole-list re-decode + view update is O(n²) and serializes the loop).
        var uiSec = 0.0
        var lastEmit = t0
        for try await step in stream {
            if stats.prefillSec == 0 {
                stats.prefillSec = Self.seconds(since: t0)
                tGen = SuspendingClock.now
                lastEmit = tGen
            }
            let tok = Int(step.tokenId)
            stats.decodeTok += 1
            if let eos, tok == eos { break }
            if tok == Int(EOT) { break }
            gen.append(tok)
            let now = SuspendingClock.now
            if now - lastEmit >= .milliseconds(40) {
                onText(gen)
                uiSec += Self.seconds(since: now)
                lastEmit = SuspendingClock.now
            }
        }
        let tFlush = SuspendingClock.now
        onText(gen)
        uiSec += Self.seconds(since: tFlush)
        stats.decodeSec = Self.seconds(since: tGen) - uiSec
        return stats
    }

    // MARK: - Preprocessing (mirrors Gemma4ImageProcessor for the square grid)

    /// 768x768 RGB resize -> /255 (graph scales 2x-1 in-graph) -> ROW-MAJOR
    /// patchify with per-patch [py, px, c] (HWC) flatten -> [2304 * 768] f16.
    static func preprocess(cgImage: CGImage) -> [Float16] {
        let S = 768, P = 16
        let grid = S / P  // 48

        var rgba = [UInt8](repeating: 0, count: S * S * 4)
        let ctx = CGContext(
            data: &rgba, width: S, height: S, bitsPerComponent: 8,
            bytesPerRow: S * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: S, height: S))

        var out = [Float16](repeating: 0, count: 2304 * 768)
        var patchIdx = 0
        for pr in 0..<grid {
            for pc in 0..<grid {
                let y0 = pr * P, x0 = pc * P
                let base = patchIdx * 768
                for py in 0..<P {
                    for px in 0..<P {
                        let o = base + (py * P + px) * 3
                        let src = ((y0 + py) * S + (x0 + px)) * 4
                        out[o] = Float16(Float(rgba[src]) / 255.0)
                        out[o + 1] = Float16(Float(rgba[src + 1]) / 255.0)
                        out[o + 2] = Float16(Float(rgba[src + 2]) / 255.0)
                    }
                }
                patchIdx += 1
            }
        }
        return out
    }

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let d = SuspendingClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "Gemma4VLBackend", code: 1,
                userInfo: [NSLocalizedDescriptionKey: msg])
    }
}

// MARK: - PLE row table (int8 per-row quant dump, mmap; provider-mode source)

final class PLERowTable: @unchecked Sendable {
    private let base: UnsafeRawPointer
    private let scales: [Float]
    private let scalePL: Float
    private let rowElems: Int

    init(rawDir: URL) throws {
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: rawDir.appendingPathComponent("meta.json"))) as! [String: Any]
        let vocab = meta["V"] as! Int
        rowElems = meta["PLD"] as! Int
        scalePL = Float((meta["embed_scale_pl"] as! NSNumber).doubleValue)

        let scaleData = try Data(
            contentsOf: rawDir.appendingPathComponent("embed_per_layer.scale.f32"))
        scales = scaleData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        precondition(scales.count == vocab, "scale count \(scales.count) != V \(vocab)")

        let tablePath = rawDir.appendingPathComponent("embed_per_layer.i8").path
        let fd = open(tablePath, O_RDONLY)
        precondition(fd >= 0, "cannot open \(tablePath)")
        defer { close(fd) }
        let length = vocab * rowElems
        guard let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            fatalError("mmap failed for \(tablePath)")
        }
        madvise(p, length, MADV_RANDOM)
        base = UnsafeRawPointer(p)
    }

    /// ple_tokens row for `token`: q[token]·scale[token]·√ld, fp16 [1,1,L,ld].
    func fill(token: Int32, destination: UnsafeMutableRawPointer, byteCount: Int) {
        let t = Int(token)
        guard t >= 0, t < scales.count,
            byteCount == rowElems * MemoryLayout<Float16>.size
        else {
            memset(destination, 0, byteCount)
            return
        }
        let row = base.advanced(by: t * rowElems).assumingMemoryBound(to: Int8.self)
        let s = scales[t] * scalePL
        let out = destination.assumingMemoryBound(to: Float16.self)
        for i in 0..<rowElems {
            out[i] = Float16(Float(row[i]) * s)
        }
    }
}
