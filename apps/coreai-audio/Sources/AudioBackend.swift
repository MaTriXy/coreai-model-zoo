// AudioBackend — Qwen2.5-Omni Thinker audio UNDERSTANDING on the pipelined engine (the zoo's
// first audio model). Mirrors Qwen3VLBackend but is the simpler rider: ONE static graph input
// (audio_embeds [750,2048]) and NO rope shift (TMRoPE collapses to 1-D → engine-native positions).
//
// Two models under Documents/models/:
//   * decoder LanguageBundle (S=1 static query, int8lin) — the audio embeds ride one static buffer;
//     <|AUDIO|> ids are REWRITTEN to extension ids vocab+slot, the graph gathers audio_embeds[id-V].
//     iOS uses the AOT .aimodelc (dodges the 3.9 GB on-device JIT jetsam); macOS uses the JIT .aimodel.
//   * audio encoder .aimodel (fixed K=15 Whisper tower), run ONCE per clip:
//     input_features [1,128,3000] + attn_bias [15,1,1,100] -> audio_embeds [750,2048].
//
// Host front end (Swift vDSP, bit-exact vs the HF extractor, gated cos 1.0): waveform → log-mel →
// pad to K chunks. See QWEN2_5_OMNI_THINKER_STATE.md.

import Accelerate
import CoreAI
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Metal
import Tokenizers

@MainActor
final class AudioBackend: ObservableObject {
    @Published var status = "Tap Load."
    @Published var answer = ""
    @Published var loaded = false
    @Published var busy = false

    private let arch = AudioArchitecture.qwen2_5Omni3B

    // Platform-specific decoder bundle (iOS = AOT .aimodelc, macOS = JIT .aimodel); encoder shared.
    #if os(iOS)
        private let decoderDir = "qwen2_5_omni_3b_thinker_n750_ios"
    #else
        private let decoderDir = "qwen2_5_omni_3b_thinker_int8lin_n750_s1_bundle"
    #endif
    private let encoderDir = "qwen2_5_omni_3b_audio_encoder_fp16_k15.aimodel"

    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    private var encoderModel: AIModel?
    private var encoderFn: InferenceFunction?
    private var encoderD: InferenceFunctionDescriptor?
    private var mel: AudioMelPreprocessor?
    private var audioBuf: MTLBuffer?        // [750,2048] f16 static input
    private var attachedTokens = 0
    private var ctx = 4096

    func load() async {
        guard !loaded, !busy else { return }
        busy = true; status = "Loading audio model…"
        do {
            if getenv("COREAI_CHUNK_THRESHOLD") == nil { setenv("COREAI_CHUNK_THRESHOLD", "1", 1) }
            let models = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("models")

            guard let device = MTLCreateSystemDefaultDevice() else { throw Self.err("no Metal device") }
            let bytes = arch.audioEmbedsCount * 2
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw Self.err("audio buffer alloc failed")
            }
            memset(buf.contents(), 0, bytes)
            audioBuf = buf

            let bundle = try LanguageBundle(at: models.appendingPathComponent(decoderDir))
            ctx = bundle.maxContextLength
            let config = ModelConfig(
                name: bundle.name, tokenizer: bundle.tokenizer, vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength, serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main")
            engine = try await EngineFactory.createEngine(
                config: try JSONEncoder().encode(config),
                modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
                options: EngineOptions(staticInputBuffers: ["audio_embeds": StaticInputBuffer(buf)]))
            tokenizer = try await bundle.loadTokenizer()

            // Audio encoder (plain .aimodel, GPU). resolveAsset takes .aimodel or .aimodelc.
            let encURL = models.appendingPathComponent(encoderDir)
                .appendingPathComponent(encoderDir)
            var eo = SpecializationOptions(preferredComputeUnitKind: .gpu)
            eo.expectFrequentReshapes = false
            let em = try await AIModel(contentsOf: encURL, options: eo)
            guard let fn = try em.loadFunction(named: "main") else { throw Self.err("encoder main missing") }
            encoderModel = em; encoderFn = fn; encoderD = fn.descriptor
            mel = try AudioMelPreprocessor.qwen2_5Omni()

            _ = try await runDecoder(ids: [9707], maxTokens: 1, eos: nil, onText: { _ in })  // warmup
            loaded = true
            status = "Model ready (\(availableMB()) MB free). Record, choose, or demo, then Ask."
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        busy = false
    }

    // MARK: - Attach audio (mel → encoder → static buffer)

    /// Encode a 16 kHz mono waveform into the decoder's audio buffer.
    func attach(samples: [Float]) async throws {
        guard let mel, let encoderFn, let encoderD, let audioBuf else { throw Self.err("not loaded") }
        let maxSamples = arch.melFrames * 160  // ≈30 s
        let clip = samples.count > maxSamples ? Array(samples[0..<maxSamples]) : samples
        let (logmel, frames) = mel.logMel(clip)
        let (feats, bias, n) = arch.encoderInputs(fromMel: logmel, frames: frames)
        guard n > 0, n <= arch.maxAudioTokens else { throw Self.err("clip out of range (\(n) tokens)") }

        guard case .ndArray(let fin)? = encoderD.inputDescriptor(of: "input_features"),
              case .ndArray(let bin)? = encoderD.inputDescriptor(of: "attn_bias") else {
            throw Self.err("encoder inputs missing")
        }
        var fArr = NDArray(descriptor: fin.resolvingDynamicDimensions([1, arch.melBins, arch.melFrames]))
        fillNDArray(&fArr, as: Float16.self, with: feats)
        var bArr = NDArray(descriptor: bin.resolvingDynamicDimensions([arch.chunks, 1, 1, arch.headsFrames]))
        fillNDArray(&bArr, as: Float16.self, with: bias)

        guard case .ndArray(let eout)? = encoderD.outputDescriptor(of: "audio_embeds") else {
            throw Self.err("encoder output missing")
        }
        var embOut = NDArray(descriptor: eout.resolvingDynamicDimensions([arch.maxAudioTokens, arch.hidden]))
        var out = InferenceFunction.MutableViews()
        out.insert(&embOut, for: "audio_embeds")
        _ = try await encoderFn.run(
            inputs: ["input_features": fArr, "attn_bias": bArr],
            states: InferenceFunction.MutableViews(), outputViews: consume out)

        let emb = flattenAsFloat(embOut)
        memset(audioBuf.contents(), 0, audioBuf.length)
        let p = audioBuf.contents().assumingMemoryBound(to: Float16.self)
        let valid = min(n * arch.hidden, emb.count)
        for i in 0..<valid { p[i] = Float16(emb[i]) }
        attachedTokens = n
    }

    // MARK: - Generate

    func ask(_ question: String, maxNew: Int = 128) async {
        guard let tokenizer, attachedTokens > 0, !busy else { return }
        busy = true; answer = ""; status = "Thinking… (\(availableMB()) MB free)"
        do {
            let q = question.isEmpty ? "What do you hear?" : question
            let text = "\(arch.imStart)system\nYou are a helpful assistant.\(arch.imEnd)\n"
                + "\(arch.imStart)user\n\(arch.audioBos)"
                + String(repeating: arch.audioPad, count: attachedTokens)
                + "\(arch.audioEos)\(q)\(arch.imEnd)\n\(arch.imStart)assistant\n"
            var ids = tokenizer.encode(text: text).map { Int32($0) }
            let pad = tokenizer.encode(text: arch.audioPad)
            guard pad.count == 1 else { throw Self.err("\(arch.audioPad) not a single token") }
            let padId = Int32(pad[0])
            var slot: Int32 = 0
            for i in 0..<ids.count where ids[i] == padId && slot < Int32(attachedTokens) {
                ids[i] = arch.vocab + slot; slot += 1
            }
            guard slot == Int32(attachedTokens) else { throw Self.err("audio pad rewrite mismatch") }
            guard ids.count < ctx - 1 else { throw Self.err("prompt does not fit ctx") }

            _ = try await runDecoder(ids: ids, maxTokens: min(maxNew, ctx - ids.count - 1),
                                     eos: tokenizer.eosTokenId) { gen in
                self.answer = tokenizer.decode(tokens: gen, skipSpecialTokens: true)
            }
            status = "Done (\(availableMB()) MB free)."
        } catch {
            status = "Generation failed: \(error.localizedDescription)"
        }
        busy = false
    }

    private func runDecoder(ids: [Int32], maxTokens: Int, eos: Int?, onText: ([Int]) -> Void) async throws {
        guard let engine else { throw Self.err("not loaded") }
        try await engine.reset()
        let stream = try engine.generate(
            with: ids, samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        var gen: [Int] = []
        for try await step in stream {
            let tok = Int(step.tokenId)
            if let eos, tok == eos { break }
            if tok == 151_645 { break }  // <|im_end|>
            gen.append(tok)
            onText(gen)
        }
    }

    private func availableMB() -> Int {
        #if os(iOS)
            return Int(os_proc_available_memory() / (1024 * 1024))
        #else
            return 0
        #endif
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "AudioBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
