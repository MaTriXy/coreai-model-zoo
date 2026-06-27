// Qwen3VLBackend — Qwen3-VL 2B / 4B on the pipelined engine (the zoo's first
// VLM). Both share the architecture (fixed-grid ViT, 3 deepstack levels); only
// the decoder hidden size differs (2B 2048 / 4B 2560), carried in `Spec`.
//
// Two models under Documents/models/:
//   * qwen3_vl_2b_instruct_decode_int8hu_s1 — the text decoder LanguageBundle
//     (S=1 static query; int8 body + absmax int8 head). Multimodal state rides
//     4 static graph inputs (engine static-inputs patch): image_embeds
//     [196,2048] f16, deepstack_embeds [588,2048] f16, rope_shift_start [1]
//     i32, rope_shift_amount [1] i32 — owned MTLBuffers this class rewrites
//     per attached image (cheap: ~3.2 MB total).
//   * qwen3_vl_2b_instruct_vision — the fixed-grid ViT .aimodel, run ONCE per
//     image: patches [784,1536] f16 -> (image_embeds, deepstack_embeds).
//
// Host contract (mirrors the gated python pipeline):
//   * preprocess: resize 448x448, x/127.5-1, patchify block-major, per-patch
//     [C,T,P,P] flatten with the frame duplicated (temporal_patch_size 2).
//   * prompt: ChatML with <|vision_start|> + 196x<|image_pad|> + <|vision_end|>
//     before the user text; image_pad ids REWRITTEN to extension ids V+slot.
//   * rope shift: start = imgStart + 196, amount = 196 - 14 = 182; text-only
//     turns bind 1<<30 / 0 (the graph degenerates to a plain Qwen3 LLM).
//
// Numerics: Mac gates A/B/D vs fp32-HF PASS; iPhone 24/24 both prompts x3
// (PipelinedBench). See QWEN3VL_STATE.md.

import CoreAI
import CoreAILanguageModels
import CoreAIShared
import CoreGraphics
import Foundation
import Metal
import Tokenizers

@MainActor
final class Qwen3VLBackend {
    // Per-model wiring. 2B and 4B share the fixed-grid ViT (depth 24, 3 deepstack
    // levels, vocab 151936); only the decoder hidden size differs, which sizes the
    // image_embeds / deepstack_embeds buffers.
    struct Spec: Sendable {
        let decoderBundle: String   // dir under Documents/models/
        let visionDir: String       // dir holding <visionDir>.aimodel
        let hfDecoderPath: String   // subpath inside the HF repo
        let hfVisionPath: String
        let label: String
        let hid: Int                // decoder hidden size (2B 2048 / 4B 2560)
    }
    nonisolated static let qwen3vl2b = Spec(
        decoderBundle: "qwen3_vl_2b_instruct_decode_int8hu_s1",
        visionDir: "qwen3_vl_2b_instruct_vision",
        hfDecoderPath: "gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1",
        hfVisionPath: "gpu-pipelined/qwen3_vl_2b_instruct_vision",
        label: "Qwen3-VL 2B ⚡pipelined", hid: 2048)
    nonisolated static let qwen3vl4b = Spec(
        decoderBundle: "qwen3_vl_4b_instruct_decode_int8hu_s1",
        visionDir: "qwen3_vl_4b_instruct_vision",
        hfDecoderPath: "gpu-pipelined/qwen3_vl_4b_instruct_decode_int8hu_s1",
        hfVisionPath: "gpu-pipelined/qwen3_vl_4b_instruct_vision",
        label: "Qwen3-VL 4B ⚡pipelined", hid: 2560)
    // Holo2-4B (H Company GUI-grounding / computer-use VLM) — Qwen3-VL-4B backbone,
    // rides this same backend. int8lin body (vs the qwen *_int8hu* ship config).
    nonisolated static let holo2 = Spec(
        decoderBundle: "holo2_4b_decode_int8lin_s1",
        visionDir: "holo2_4b_vision",
        hfDecoderPath: "gpu-pipelined/holo2_4b_decode_int8lin_s1",
        hfVisionPath: "gpu-pipelined/holo2_4b_vision",
        label: "Holo2-4B ⚡pipelined", hid: 2560)

    let spec: Spec
    init(spec: Spec) { self.spec = spec }

    // Architecture constants (448x448 grid; shared by 2B/4B)
    private let V: Int32 = 151_936
    private let N = 196          // merged vision tokens (14x14)
    private let GRID = 14
    private var HID: Int { spec.hid }
    private let PATCHES = 784    // 28x28 pre-merge
    private let PATCH_DIM = 1536 // 3 * 2 * 16 * 16

    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    private var visionModel: AIModel?
    private var visionFn: InferenceFunction?
    private var visionD: InferenceFunctionDescriptor?
    private(set) var ctx = 4096

    // Owned static-input buffers (alive for the engine's lifetime)
    private var imgBuf: MTLBuffer?
    private var dsBuf: MTLBuffer?
    private var shiftStartBuf: MTLBuffer?
    private var shiftAmountBuf: MTLBuffer?

    private(set) var imageAttached = false

    var loaded: Bool { engine != nil }

    func load() async throws {
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let models = docs.appendingPathComponent("models")
        let bundle = try LanguageBundle(at: models.appendingPathComponent(spec.decoderBundle))
        ctx = bundle.maxContextLength

        guard let device = MTLCreateSystemDefaultDevice() else { throw Self.err("no Metal device") }
        func owned(_ bytes: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            memset(b.contents(), 0, bytes)
            return b
        }
        let img = owned(N * HID * 2)
        let ds = owned(3 * N * HID * 2)
        let ss = owned(64)  // engine pads [1] i32 static inputs to 64 bytes
        let sa = owned(64)
        imgBuf = img; dsBuf = ds; shiftStartBuf = ss; shiftAmountBuf = sa
        setTextOnlyShift()

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
            options: EngineOptions(staticInputBuffers: [
                "image_embeds": StaticInputBuffer(img),
                "deepstack_embeds": StaticInputBuffer(ds),
                "rope_shift_start": StaticInputBuffer(ss),
                "rope_shift_amount": StaticInputBuffer(sa),
            ])
        )
        tokenizer = try await bundle.loadTokenizer()

        // vision tower (plain .aimodel, GPU)
        let visURL = models.appendingPathComponent(spec.visionDir)
            .appendingPathComponent("\(spec.visionDir).aimodel")
        guard FileManager.default.fileExists(atPath: visURL.path) else {
            throw Self.err("missing \(spec.visionDir).aimodel in Documents/models/\(spec.visionDir)")
        }
        var vo = SpecializationOptions(preferredComputeUnitKind: .gpu)
        vo.expectFrequentReshapes = false
        let vm = try await AIModel(contentsOf: visURL, options: vo)
        guard let fn = try vm.loadFunction(named: "main") else { throw Self.err("vision main missing") }
        visionModel = vm
        visionFn = fn
        visionD = fn.descriptor

        // 1-token warmup through the decoder graph
        _ = try await run(ids: [9707], maxTokens: 1, eos: nil, onText: { _ in })
        print(PipelinedBackend.memLine("\(spec.label) loaded"))
    }

    func unload() {
        engine = nil
        tokenizer = nil
        visionFn = nil
        visionModel = nil
        imgBuf = nil; dsBuf = nil; shiftStartBuf = nil; shiftAmountBuf = nil
        imageAttached = false
    }

    // MARK: - Image attach

    /// Preprocess + vision-encode `cgImage` and write the embeds into the
    /// decoder's static-input buffers. Stays attached across turns until
    /// replaced (each generate re-prefills, so one image serves a whole chat).
    func attach(cgImage: CGImage) async throws {
        guard let visionFn, let visionD, let imgBuf, let dsBuf else {
            throw Self.err("backend not loaded")
        }
        let t0 = SuspendingClock.now
        let patches = Self.preprocess(cgImage: cgImage)

        guard case .ndArray(let pin)? = visionD.inputDescriptor(of: "patches") else {
            throw Self.err("patches input missing")
        }
        var pArr = NDArray(descriptor: pin.resolvingDynamicDimensions([PATCHES, PATCH_DIM]))
        fillNDArray(&pArr, as: Float16.self, with: patches)

        guard case .ndArray(let e0)? = visionD.outputDescriptor(of: "image_embeds"),
              case .ndArray(let e1)? = visionD.outputDescriptor(of: "deepstack_embeds") else {
            throw Self.err("vision outputs missing")
        }
        var embOut = NDArray(descriptor: e0.resolvingDynamicDimensions([N, HID]))
        var dsOut = NDArray(descriptor: e1.resolvingDynamicDimensions([3 * N, HID]))
        var out = InferenceFunction.MutableViews()
        out.insert(&embOut, for: "image_embeds")
        out.insert(&dsOut, for: "deepstack_embeds")
        _ = try await visionFn.run(
            inputs: ["patches": pArr],
            states: InferenceFunction.MutableViews(),
            outputViews: consume out)

        let emb = flattenAsFloat(embOut)
        let dsf = flattenAsFloat(dsOut)
        let imgP = imgBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<emb.count { imgP[i] = Float16(emb[i]) }
        let dsP = dsBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<dsf.count { dsP[i] = Float16(dsf[i]) }
        imageAttached = true
        let dt = Self.seconds(since: t0)
        print(String(format: "[qwen3vl] image attached (vision %.0f ms)", dt * 1000))
    }

    func detachImage() {
        imageAttached = false
        if let imgBuf { memset(imgBuf.contents(), 0, imgBuf.length) }
        if let dsBuf { memset(dsBuf.contents(), 0, dsBuf.length) }
        setTextOnlyShift()
    }

    private func setTextOnlyShift() {
        shiftStartBuf?.contents().assumingMemoryBound(to: Int32.self)[0] = 1 << 30
        shiftAmountBuf?.contents().assumingMemoryBound(to: Int32.self)[0] = 0
    }

    // MARK: - Generation

    func generate(_ prompt: String, maxNew: Int,
                  onUpdate: (String) -> Void) async throws -> PipelinedBackend.GenStats {
        guard let tokenizer else { throw Self.err("backend not loaded") }
        var ids: [Int32]
        if imageAttached {
            // ChatML with the vision block; encode then rewrite pads in one pass.
            let text = "<|im_start|>user\n<|vision_start|>"
                + String(repeating: "<|image_pad|>", count: N)
                + "<|vision_end|>\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(text: text).map { Int32($0) }
            guard let padId = firstPadId(ids) else { throw Self.err("image pads not encoded") }
            var slot: Int32 = 0
            var imgStart = -1
            for i in 0..<ids.count where ids[i] == padId {
                if imgStart < 0 { imgStart = i }
                ids[i] = V + slot
                slot += 1
            }
            guard slot == Int32(N), imgStart >= 0 else {
                throw Self.err("expected \(N) image pads, found \(slot)")
            }
            shiftStartBuf?.contents().assumingMemoryBound(to: Int32.self)[0] =
                Int32(imgStart + N)
            shiftAmountBuf?.contents().assumingMemoryBound(to: Int32.self)[0] =
                Int32(N - GRID)
        } else {
            setTextOnlyShift()
            if let t = try? tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]]) {
                ids = t.map { Int32($0) }
            } else {
                ids = tokenizer.encode(
                    text: "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
                ).map { Int32($0) }
            }
        }
        guard ids.count < ctx - 1 else {
            throw Self.err("prompt (\(ids.count) tok) does not fit ctx \(ctx)")
        }
        let budget = min(maxNew, ctx - ids.count - 1)
        let stats = try await run(ids: ids, maxTokens: budget, eos: tokenizer.eosTokenId) { gen in
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
        print(PipelinedBackend.memLine("\(spec.label) gen"))
        return stats
    }

    private func firstPadId(_ ids: [Int32]) -> Int32? {
        guard let tokenizer else { return nil }
        let pad = tokenizer.encode(text: "<|image_pad|>")
        return pad.count == 1 ? Int32(pad[0]) : nil
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
        var stats = PipelinedBackend.GenStats(label: spec.label, prefillTok: ids.count)
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
            if tok == 151_645 { break }  // <|im_end|>
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

    // MARK: - Preprocessing (mirrors Qwen2VLImageProcessor for the fixed grid)

    /// 448x448 RGB resize -> x/127.5-1 -> block-major patchify, per-patch
    /// [C, T(dup), 16, 16] flatten -> [784 * 1536] f16.
    static func preprocess(cgImage: CGImage) -> [Float16] {
        let S = 448, P = 16, M = 2
        let gridP = S / P          // 28 patches per side
        let gridB = gridP / M      // 14 blocks per side

        var rgba = [UInt8](repeating: 0, count: S * S * 4)
        let ctx = CGContext(
            data: &rgba, width: S, height: S, bitsPerComponent: 8,
            bytesPerRow: S * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: S, height: S))

        func px(_ x: Int, _ y: Int, _ c: Int) -> Float16 {
            Float16(Float(rgba[(y * S + x) * 4 + c]) / 127.5 - 1.0)
        }

        var out = [Float16](repeating: 0, count: 784 * 1536)
        var patchIdx = 0
        for br in 0..<gridB {
            for bc in 0..<gridB {
                for ir in 0..<M {
                    for ic in 0..<M {
                        let pr = br * M + ir, pc = bc * M + ic
                        let y0 = pr * P, x0 = pc * P
                        let base = patchIdx * 1536
                        for c in 0..<3 {
                            for t in 0..<2 {           // duplicated frame
                                let cBase = base + (c * 2 + t) * P * P
                                for py in 0..<P {
                                    for pxi in 0..<P {
                                        out[cBase + py * P + pxi] = px(x0 + pxi, y0 + py, c)
                                    }
                                }
                            }
                        }
                        patchIdx += 1
                    }
                }
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
        NSError(domain: "Qwen3VLBackend", code: 1,
                userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
