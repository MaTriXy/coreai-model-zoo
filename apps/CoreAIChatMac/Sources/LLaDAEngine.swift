// LLaDAEngine — drives the zoo's first DIFFUSION LLM (LLaDA-8B / d3LLM) in CoreAIChatMac.
//
// Unlike the AR models (Apple's coreai-sequential InferenceEngine streaming one token at a time),
// a masked-diffusion LM is a HOST loop over a single static, BIDIRECTIONAL forward:
//   bundle fn  main(input_ids[1,S] Int32) -> logits[1,S,VOCAB] Float16   (full canvas, NO KV cache)
//   canvas = prompt ++ [MASK]*gen ; each step: forward -> unmask the lowest-ENTROPY masked positions
//   in the active semi-AR block (always ≥1, plus any below `threshold`) -> repeat block by block.
// This is the validated loop from conversion/dllm/generate_llada.py / ondevice/LLaDARunner, adapted
// to stream the canvas as it denoises (a genuinely different, parallel "fill-in" UX vs left-to-right).

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

@MainActor
final class LLaDAEngine {
    struct Params {
        var seq = 128
        var blockSize = 32
        var threshold = 0.5
        var maskTokenId: Int32 = 126336
        var eosTokenId: Int32 = 126081
    }

    private let model: AIModel
    private let fn: InferenceFunction
    private let desc: InferenceFunctionDescriptor
    private let tokenizer: any Tokenizer
    let params: Params
    let vocab: Int

    // Whether a bundle directory is a diffusion LM (vs an AR LanguageBundle) — cheap metadata peek.
    static func isDLLM(bundleURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: bundleURL.appendingPathComponent("metadata.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return (obj["kind"] as? String) == "dllm"
    }

    init(bundleURL: URL) async throws {
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: bundleURL.appendingPathComponent("metadata.json"))) as? [String: Any] ?? [:]
        let assets = meta["assets"] as? [String: Any]
        let assetName = (assets?["main"] as? String) ?? "main.aimodel"
        var p = Params()
        if let d = meta["diffusion"] as? [String: Any] {
            if let v = d["seq"] as? Int { p.seq = v }
            if let v = d["block_size"] as? Int { p.blockSize = v }
            if let v = d["threshold"] as? Double { p.threshold = v }
            if let v = d["mask_token_id"] as? Int { p.maskTokenId = Int32(v) }
            if let v = d["eos_token_id"] as? Int { p.eosTokenId = Int32(v) }
        }
        self.params = p

        var opts = SpecializationOptions(preferredComputeUnitKind: .gpu)
        let assetURL = bundleURL.appendingPathComponent(assetName)
        self.model = try await AIModel(contentsOf: assetURL, options: opts)
        guard let d = model.functionDescriptor(for: "main"), let f = try model.loadFunction(named: "main") else {
            throw NSError(domain: "LLaDAEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "bundle has no 'main' function"])
        }
        self.desc = d
        self.fn = f
        guard case .ndArray(let nd)? = d.outputDescriptor(of: "logits") else {
            throw NSError(domain: "LLaDAEngine", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "no logits output"])
        }
        self.vocab = nd.shape.last.map { $0 < 0 ? 0 : $0 } ?? 0

        self.tokenizer = try await AutoTokenizer.from(modelFolder: bundleURL.appendingPathComponent("tokenizer"))
    }

    // MARK: - forward

    private func alloc(_ name: String, _ shape: [Int], input: Bool) -> NDArray {
        let io = input ? desc.inputDescriptor(of: name) : desc.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(shape))
    }

    private func forward(_ canvas: [Int32]) async throws -> [Float] {
        let S = params.seq
        var idA = alloc("input_ids", [1, S], input: true)
        fillNDArray(&idA, as: Int32.self, with: canvas)
        var lg = alloc("logits", [1, S, vocab], input: false)
        var ov = InferenceFunction.MutableViews(); ov.insert(&lg, for: "logits")
        _ = try await fn.run(inputs: ["input_ids": idA],
                             states: InferenceFunction.MutableViews(), outputViews: consume ov)
        return flattenAsFloat(lg)
    }

    // (argmax token, entropy in nats) over one logits row of length `vocab`. fp64 entropy, matches Python.
    private func rowEntropyArgmax(_ logits: [Float], _ base: Int) -> (Int32, Double) {
        var maxv = -Float.greatestFiniteMagnitude, argmax = 0
        for j in 0..<vocab { let v = logits[base + j]; if v > maxv { maxv = v; argmax = j } }
        var sumExp = 0.0
        for j in 0..<vocab { sumExp += Foundation.exp(Double(logits[base + j] - maxv)) }
        let logZ = Foundation.log(sumExp)
        var ent = 0.0
        for j in 0..<vocab {
            let z = Double(logits[base + j] - maxv)
            let p = Foundation.exp(z) / sumExp
            if p > 0 { ent -= p * (z - logZ) }
        }
        return (Int32(argmax), ent)
    }

    // MARK: - prompt

    func promptIds(history: [[String: any Sendable]]) throws -> [Int32] {
        let ids = try tokenizer.applyChatTemplate(messages: history)
        return ids.map(Int32.init)
    }

    // Decode the committed (non-mask) generated tokens to clean readable text (final answer).
    func decodeGenerated(_ canvas: [Int32], promptLen: Int) -> String {
        var toks: [Int] = []
        for i in promptLen..<canvas.count {
            let t = canvas[i]
            if t == params.eosTokenId { break }
            if t == params.maskTokenId { continue }
            toks.append(Int(t))
        }
        return tokenizer.decode(tokens: toks)
    }

    // LIVE diffusion view: render the active region [promptLen, frontier) with still-masked positions
    // shown as ░ — so the user watches the canvas denoise (tokens pop in IN PARALLEL, out of reading
    // order), the way a diffusion LM actually decodes. Committed runs are decoded; mask runs become ░.
    func renderCanvas(_ canvas: [Int32], promptLen: Int, frontier: Int) -> String {
        var parts: [String] = []
        var run: [Int] = []
        func flush() { if !run.isEmpty { parts.append(tokenizer.decode(tokens: run)); run.removeAll(keepingCapacity: true) } }
        for i in promptLen..<min(frontier, canvas.count) {
            let t = canvas[i]
            if t == params.eosTokenId { break }
            if t == params.maskTokenId { flush(); parts.append("░") }
            else { run.append(Int(t)) }
        }
        flush()
        return parts.joined()
    }

    // MARK: - denoising loop

    struct StepInfo { let text: String; let committed: Int; let nfe: Int }

    /// Run the diffusion decode. `onStep` is called after every denoising step with the current
    /// decoded text (for live streaming). Returns (finalText, nfe, committedTokens).
    func generate(history: [[String: any Sendable]],
                  onStep: @MainActor (StepInfo) -> Void) async throws -> (text: String, nfe: Int, committed: Int) {
        let S = params.seq, BLOCK = params.blockSize, MASK = params.maskTokenId
        // Fit the prompt inside the fixed canvas while RESERVING room to generate. The canvas has no
        // KV cache, so the whole prompt+answer must fit in S. Drop the OLDEST turns first (keep the
        // current question); never let generation collapse to a sliver (that yields broken 1-token
        // answers). At small S this effectively makes each turn standalone; larger S keeps more memory.
        let promptCap = max(8, S / 2)
        var msgs = history
        var ids = try promptIds(history: msgs)
        while ids.count > promptCap && msgs.count > 1 {
            msgs.removeFirst()
            ids = try promptIds(history: msgs)
        }
        if ids.count > S - 1 { ids = Array(ids.suffix((S * 3) / 4)) }   // pathological single huge message
        let L = ids.count
        let GEN = S - L

        var canvas = [Int32](repeating: MASK, count: S)
        for i in 0..<L { canvas[i] = ids[i] }

        let numBlocks = (GEN + BLOCK - 1) / BLOCK
        var nfe = 0

        blockLoop: for nb in 0..<numBlocks {
            let blkLo = L + nb * BLOCK
            let blkHi = min(L + (nb + 1) * BLOCK, S)
            while true {
                nfe += 1
                let logits = try await forward(canvas)

                var cands: [(pos: Int, tok: Int32, ent: Double)] = []
                for i in 0..<blkHi where canvas[i] == MASK {
                    let (tok, ent) = rowEntropyArgmax(logits, i * vocab)
                    cands.append((i, tok, ent))
                }
                if cands.isEmpty { break }
                cands.sort { $0.ent < $1.ent }
                for (r, c) in cands.enumerated() where r == 0 || c.ent <= params.threshold {
                    canvas[c.pos] = c.tok
                }

                let committed = (L..<S).reduce(0) { canvas[$1] != MASK ? $0 + 1 : $0 }
                // live diffusion view: active region with ░ for positions still being denoised
                let text = renderCanvas(canvas, promptLen: L, frontier: blkHi)
                onStep(StepInfo(text: text, committed: committed, nfe: nfe))
                await Task.yield()   // let SwiftUI render the streamed canvas

                var blkMasks = 0
                for i in blkLo..<blkHi where canvas[i] == MASK { blkMasks += 1 }
                if blkMasks == 0 { break }
                if nfe > 4096 { break blockLoop }
            }
            if (L..<blkHi).contains(where: { canvas[$0] == params.eosTokenId }) { break }
        }

        let committed = (L..<S).reduce(0) { canvas[$1] != MASK ? $0 + 1 : $0 }
        return (decodeGenerated(canvas, promptLen: L), nfe, committed)
    }
}
