// OCRPipeline.swift — Unlimited-OCR (baidu) on-device document OCR via Core AI.
//
// Drives the stock coreai.runtime DIRECTLY (no engine patch): a stateless vision
// encoder (CoreAIKitVision.GraphModel) + a stateful, inputs_embeds-driven decoder
// (CoreAI.AIModel with the unified "prefill"/"decode" functions, KV-cache state via
// InferenceFunction.MutableViews — the CoreAISequentialEngine pattern).
//
// The whole pipeline mirrors the verified Python reference (_unlimited_ocr/
// _ocr_pipeline.py); the recipe + constants are in out/_swift_assets/recipe.json.
//
//   CGImage --preprocess(640, norm .5)--> [1,3,640,640]
//     --vision .aimodel--> visual tokens [1,100,1280]
//     --arrange (10x10 + image_newline rows + view_seperator)--> [111,1280]
//     --scatter into embed_tokens(prompt_input_ids)--> prefix [1,115,1280]
//     --decoder prefill(prefix) + greedy decode loop--> token ids --detok--> markdown
//
// NOTE: built for macOS 27+, Swift 6. Needs an on-device build (the harness can't
// compile/run macOS apps). Asset files (out/_swift_assets/* + the two .aimodel
// bundles) ship alongside or download from HF; see ModelAssets below.

import CoreAI
import CoreAIKitVision
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers  // swift-transformers

// MARK: - Locked spec (from the verified recipe)

enum OCRSpec {
    static let imageSide = 640          // image_size; crop_mode = false (Base mode)
    static let patchGrid = 10           // 10x10 = 100 patches
    static let visualTokens = 111       // 100 patches + 10 row newlines + 1 seperator
    static let prefixLen = 115          // 111 visual + BOS + 3 prompt-text tokens
    static let hidden = 1280
    static let vocab = 129_280
    static let imageTokenID: Int32 = 128_815
    static let eos: Int32 = 1
    static let cacheLen = 2048          // StaticKVCache buffer (decoder bundle)
    static let layers = 12
    static let kvHeads = 10
    static let headDim = 128
}

// MARK: - Asset locations (bundles + raw constant tensors)

struct ModelAssets {
    let visionAIModel: URL        // _vision_export.aimodel
    let decoderAIModel: URL       // _dec_unified.aimodel (functions: prefill, decode)
    let embedTokens: URL          // embed_tokens.f16  [vocab, hidden]
    let imageNewline: URL         // image_newline.f16 [hidden]
    let viewSeperator: URL        // view_seperator.f16 [hidden]
    let promptInputIDs: URL       // prompt_input_ids.i32 [115]
    let tokenizerDir: URL         // dir containing tokenizer.json
}

// MARK: - Stateful decoder wrapper (GraphModel forbids state; this allows it)

final class StatefulDecoder: @unchecked Sendable {
    private let model: AIModel
    let prefill: InferenceFunction
    let decode: InferenceFunction
    let prefillDesc: InferenceFunctionDescriptor
    let decodeDesc: InferenceFunctionDescriptor

    init(contentsOf url: URL) async throws {
        model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let pd = model.functionDescriptor(for: "prefill"),
              let dd = model.functionDescriptor(for: "decode"),
              let pf = try model.loadFunction(named: "prefill"),
              let df = try model.loadFunction(named: "decode")
        else { throw OCRError.functionNotFound }
        prefillDesc = pd; decodeDesc = dd; prefill = pf; decode = df
    }

    /// Fresh zeroed KV cache (k,v): the bundle declares the [layers,1,kvHeads,cacheLen,
    /// headDim] fp16 static shape, so take the descriptors from the prefill function.
    func makeState() -> (k: NDArray, v: NDArray) {
        guard case .ndArray(let kd) = prefillDesc.stateDescriptor(of: prefillDesc.stateNames[0]),
              case .ndArray(let vd) = prefillDesc.stateDescriptor(of: prefillDesc.stateNames[1])
        else { fatalError("KV state descriptors missing") }
        var k = NDArray(descriptor: kd), v = NDArray(descriptor: vd)
        let n = OCRSpec.layers * OCRSpec.kvHeads * OCRSpec.cacheLen * OCRSpec.headDim
        fillNDArray(&k, as: Float16.self, count: n) { _ in 0 }
        fillNDArray(&v, as: Float16.self, count: n) { _ in 0 }
        return (k, v)
    }

    private func ndInput(_ desc: InferenceFunctionDescriptor, _ name: String,
                         _ data: [Float16], shape: [Int]) -> NDArray {
        guard case .ndArray(let d) = desc.inputDescriptor(of: name) else { fatalError("input \(name)") }
        var arr = NDArray(descriptor: d.resolvingDynamicDimensions(shape))
        fillNDArray(&arr, as: Float16.self, with: data)
        return arr
    }

    private func ndInputI32(_ desc: InferenceFunctionDescriptor, _ name: String,
                            _ data: [Int32], shape: [Int]) -> NDArray {
        guard case .ndArray(let d) = desc.inputDescriptor(of: name) else { fatalError("input \(name)") }
        var arr = NDArray(descriptor: d.resolvingDynamicDimensions(shape))
        fillNDArray(&arr, as: Int32.self, with: data)
        return arr
    }

    private func runLogits(_ fn: InferenceFunction, _ desc: InferenceFunctionDescriptor,
                           inputs: [String: NDArray], k: inout NDArray, v: inout NDArray) async throws -> [Float] {
        let keyName = desc.stateNames[0], valName = desc.stateNames[1]
        let logitsName = desc.outputNames[0]
        guard case .ndArray(let ld) = desc.outputDescriptor(of: logitsName) else { throw OCRError.badOutput }
        var logits = NDArray(descriptor: ld.resolvingDynamicDimensions([1, 1, OCRSpec.vocab]))

        var states = InferenceFunction.MutableViews()
        states.insert(&k, for: keyName)
        states.insert(&v, for: valName)
        var outViews = InferenceFunction.MutableViews()
        outViews.insert(&logits, for: logitsName)
        _ = try await fn.run(inputs: inputs, states: consume states, outputViews: consume outViews)
        // logits may be fp16 or fp32 depending on export; read as Float16 then widen.
        return readNDArray(logits, as: Float16.self, count: OCRSpec.vocab).map { Float($0) }
    }

    /// prefill the assembled prefix [115,1280] -> last-token logits; seeds k/v.
    func prefill(prefixEmbeds: [Float16], k: inout NDArray, v: inout NDArray) async throws -> [Float] {
        let inp = ndInput(prefillDesc, "inputs_embeds", prefixEmbeds, shape: [1, OCRSpec.prefixLen, OCRSpec.hidden])
        return try await runLogits(prefill, prefillDesc, inputs: ["inputs_embeds": inp], k: &k, v: &v)
    }

    /// one decode step: embedding [1280] of the current token at absolute position `pos`.
    func decode(tokenEmbed: [Float16], pos: Int32, k: inout NDArray, v: inout NDArray) async throws -> [Float] {
        let emb = ndInput(decodeDesc, "inputs_embeds", tokenEmbed, shape: [1, 1, OCRSpec.hidden])
        let p = ndInputI32(decodeDesc, "pos", [pos], shape: [1])
        return try await runLogits(decode, decodeDesc, inputs: ["inputs_embeds": emb, "pos": p], k: &k, v: &v)
    }
}

// MARK: - The pipeline

final class OCRPipeline: @unchecked Sendable {
    private let vision: GraphModel
    private let decoder: StatefulDecoder
    private let embed: [Float16]            // [vocab*hidden] embed_tokens table (host gather)
    private let imageNewline: [Float16]     // [hidden]
    private let viewSeperator: [Float16]    // [hidden]
    private let promptIDs: [Int32]          // [115]
    private let tokenizer: Tokenizer

    init(assets: ModelAssets) async throws {
        vision = try await GraphModel(contentsOf: assets.visionAIModel, computeUnits: .gpu)
        decoder = try await StatefulDecoder(contentsOf: assets.decoderAIModel)
        embed = Self.readF16(assets.embedTokens)
        imageNewline = Self.readF16(assets.imageNewline)
        viewSeperator = Self.readF16(assets.viewSeperator)
        promptIDs = Self.readI32(assets.promptInputIDs)
        tokenizer = try await AutoTokenizer.from(modelFolder: assets.tokenizerDir)
    }

    /// Run OCR on an image; returns the structured markdown (special tokens kept).
    func recognize(_ image: CGImage, maxTokens: Int = 1024) async throws -> String {
        // 1) preprocess -> [1,3,640,640] fp16, normalized mean=std=0.5
        let pixels = Preprocess.toCHW640(image)

        // 2) vision encoder -> [1,100,1280]
        let vout = try await vision.run(["image": .float16(pixels, shape: [1, 3, OCRSpec.imageSide, OCRSpec.imageSide])])
        guard let patches = vout["visual_tokens"]?.floats() else { throw OCRError.badOutput }  // 100*1280

        // 3) arrange -> 111*1280 (row-major: per row r: 10 patches then image_newline; then view_seperator)
        let H = OCRSpec.patchGrid, hid = OCRSpec.hidden
        var visual = [Float16](); visual.reserveCapacity(OCRSpec.visualTokens * hid)
        for r in 0..<H {
            for c in 0..<H {
                let base = (r * H + c) * hid
                for i in 0..<hid { visual.append(Float16(patches[base + i])) }
            }
            visual.append(contentsOf: imageNewline)   // newline after each row
        }
        visual.append(contentsOf: viewSeperator)

        // 4) prefix = embed_tokens(promptIDs); scatter the 111 visual features into <image> slots
        var prefix = [Float16](repeating: 0, count: OCRSpec.prefixLen * hid)
        var slot = 0
        for (pos, id) in promptIDs.enumerated() {
            if id == OCRSpec.imageTokenID {
                let src = slot * hid
                for i in 0..<hid { prefix[pos * hid + i] = visual[src + i] }
                slot += 1
            } else {
                let src = Int(id) * hid
                for i in 0..<hid { prefix[pos * hid + i] = embed[src + i] }
            }
        }

        // 5) decode: prefill + greedy with no_repeat_ngram + consecutive-repeat guard
        // (pure greedy derails into degenerate repeats — e.g. long "↑↑↑" runs — on dense
        // tables; the oracle used no_repeat_ngram=35. Match it + cap consecutive repeats.)
        var (k, v) = decoder.makeState()
        var logits = try await decoder.prefill(prefixEmbeds: prefix, k: &k, v: &v)
        var generated = [Int32]()
        var tok = pick(logits, generated)
        var p = Int32(OCRSpec.prefixLen)
        for _ in 0..<maxTokens {
            generated.append(tok)
            if tok == OCRSpec.eos { break }
            let tokEmbed = Array(embed[Int(tok) * hid ..< (Int(tok) + 1) * hid])
            logits = try await decoder.decode(tokenEmbed: tokEmbed, pos: p, k: &k, v: &v)
            tok = pick(logits, generated)
            p += 1
        }

        // 6) detokenize (keep special tokens so <table>/<tr>/<td> structure survives)
        let ids = generated.filter { $0 != OCRSpec.eos }.map { Int($0) }
        return tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    private let noRepeatN = 35
    private let maxRun = 6  // ban a token that would be the (maxRun+1)-th identical in a row

    /// argmax with banned tokens (no_repeat_ngram + consecutive-run cap); O(V) per step.
    private func pick(_ logits: [Float], _ gen: [Int32]) -> Int32 {
        var banned = Set<Int32>()
        // consecutive-run guard
        if gen.count >= maxRun, let last = gen.last, gen.suffix(maxRun).allSatisfy({ $0 == last }) {
            banned.insert(last)
        }
        // no_repeat_ngram: ban tokens that would repeat an existing noRepeatN-gram
        let m = noRepeatN - 1
        if gen.count >= m {
            let prefix = Array(gen.suffix(m))
            var i = 0
            while i + m < gen.count {
                if Array(gen[i..<i + m]) == prefix { banned.insert(gen[i + m]) }
                i += 1
            }
        }
        if banned.isEmpty { return Int32(argmax(logits)) }
        var best = -1; var bv = -Float.greatestFiniteMagnitude
        for i in 0..<logits.count where logits[i] > bv && !banned.contains(Int32(i)) { bv = logits[i]; best = i }
        return best >= 0 ? Int32(best) : Int32(argmax(logits))
    }

    private func argmax(_ x: [Float]) -> Int {
        var best = 0; var bv = x[0]
        for i in 1..<x.count where x[i] > bv { bv = x[i]; best = i }
        return best
    }

    private static func readF16(_ url: URL) -> [Float16] {
        let d = try! Data(contentsOf: url)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }
    private static func readI32(_ url: URL) -> [Int32] {
        let d = try! Data(contentsOf: url)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
}

enum OCRError: Error { case functionNotFound, badOutput }
