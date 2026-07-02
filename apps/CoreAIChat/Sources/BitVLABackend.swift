// BitVLABackend — zoo's first Vision-Language-ACTION model on-device. 1.58-bit ternary
// BitVLA (BitNet b1.58-2B LLM + BitSigLIP-SO400M vision, ustcwhy/BitVLA) runs image+instruction
// -> 7-DoF robot action entirely on the iPhone GPU. The 7 per-layer LLM linears use the custom
// 2-bit packed ternary Metal kernel (baked into the .aimodelc); the vision tower (ternary weights
// carried losslessly as fp16) runs once per image.
//
// Unlike the chat backends this is a LOW-LEVEL driver (no high-level text engine): we run the
// vision .aimodelc and the LLM .aimodelc directly (AIModel/InferenceFunction), build inputs_embeds
// host-side (preset-instruction text embeds + the 256 spliced vision embeds — so the phone needs
// NO tokenizer and NO 656MB embed table), and decode S=1 with our own KV-cache NDArray state.
// OpenVLA discrete action: 7 tokens, argmax over the 256-row action head -> bins -> q99 unnorm.
//
// Documents/models/:
//   bitvla_vision         — <dir>.h18p.aimodelc  (pixel_values[1,3,224,224] -> img_embeds[1,256,2560])
//   bitvla_llm_act        — <dir>.h18p.aimodelc  (inputs_embeds[1,1,2560]+position_ids -> logits[1,1,256])
//   bitvla_device_data    — e_pre.f16, e_post_<k>.f16, act_embed.f16, manifest.json

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Metal

@MainActor
final class BitVLABackend {
    static let visionDir = "bitvla_vision"
    static let llmDir = "bitvla_llm_act"
    static let dataDir = "bitvla_device_data"
    static let label = "BitVLA ⚡1.58-bit VLA"

    private let HID = 2560
    private let N_IMG = 256
    private let N_LAYERS = 30
    private let N_KV = 5
    private let HEAD_DIM = 128
    private let CAP = 384          // KV capacity (prefill ~306 + 7 gen)
    private let ACT_DIM = 7

    struct Preset { let text: String; let ePost: [Float16] /* [len*HID] */; let len: Int }
    struct DeviceData {
        var ePre: [Float16]; var ePreLen: Int
        var actEmbed: [Float16]                     // [256*HID]
        var presets: [Preset]
        var binCenters: [Float]                     // 255
        var q01: [Float]; var q99: [Float]; var mask: [Bool]
        var actLo: Int
    }

    private var visionFn: InferenceFunction?
    private var visionD: InferenceFunctionDescriptor?
    private var llmFn: InferenceFunction?
    private var llmD: InferenceFunctionDescriptor?
    private var visionModel: AIModel?
    private var llmModel: AIModel?
    private var data: DeviceData?

    var loaded: Bool { llmFn != nil }
    var presetTexts: [String] { data?.presets.map { $0.text } ?? [] }

    // MARK: - Load

    func load() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let models = docs.appendingPathComponent("models")

        let visURL = Self.modelURL(models.appendingPathComponent(Self.visionDir), Self.visionDir)
        let llmURL = Self.modelURL(models.appendingPathComponent(Self.llmDir), Self.llmDir)
        var vo = SpecializationOptions(preferredComputeUnitKind: .gpu); vo.expectFrequentReshapes = false
        // The custom ternary kernel cannot JIT on device (plain .aimodel -> "LLVM odiec" crash); it
        // must be AOT-precompiled. Load the AOT .aimodelc; expectFrequentReshapes=false (the =true
        // deferred-reshape package fails low-level load with POSIX Code=2, same as the vision tower).
        var lo = SpecializationOptions(preferredComputeUnitKind: .gpu); lo.expectFrequentReshapes = false

        print("[bitvla] loading vision \(visURL.lastPathComponent) ...")
        let vm = try await AIModel(contentsOf: visURL, options: vo)
        guard let vfn = try vm.loadFunction(named: "main") else { throw Self.err("vision main missing") }
        visionModel = vm; visionFn = vfn; visionD = vfn.descriptor
        print(PipelinedBackend.memLine("vision loaded"))

        print("[bitvla] loading LLM \(llmURL.lastPathComponent) ...")
        let lm = try await AIModel(contentsOf: llmURL, options: lo)
        guard let lfn = try lm.loadFunction(named: "main") else { throw Self.err("llm main missing") }
        llmModel = lm; llmFn = lfn; llmD = lfn.descriptor
        print(PipelinedBackend.memLine("llm loaded"))

        data = try Self.loadData(models.appendingPathComponent(Self.dataDir))
        print(PipelinedBackend.memLine("\(Self.label) loaded"))
    }

    func unload() {
        visionFn = nil; visionModel = nil; llmFn = nil; llmModel = nil; data = nil
    }

    private static func modelURL(_ dir: URL, _ name: String) -> URL {
        // Prefer the AOT <name>.h18p.aimodelc (the custom ternary kernel must be precompiled; it
        // can't JIT on device). Fall back to a plain <name>.aimodel only if no AOT package exists.
        let aot = dir.appendingPathComponent("\(name).h18p.aimodelc")
        if FileManager.default.fileExists(atPath: aot.path) { return aot }
        return dir.appendingPathComponent("\(name).aimodel")
    }

    private static func f16(_ url: URL) throws -> [Float16] {
        let d = try Data(contentsOf: url)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    private static func loadData(_ dir: URL) throws -> DeviceData {
        let mf = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent("manifest.json"))) as! [String: Any]
        let ePre = try f16(dir.appendingPathComponent("e_pre.f16"))
        let act = try f16(dir.appendingPathComponent("act_embed.f16"))
        var presets: [Preset] = []
        let plist = mf["presets"] as! [[String: Any]]
        for (k, p) in plist.enumerated() {
            let e = try f16(dir.appendingPathComponent("e_post_\(k).f16"))
            presets.append(Preset(text: p["text"] as! String, ePost: e, len: p["e_post_len"] as! Int))
        }
        func farr(_ k: String) -> [Float] { (mf[k] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) } }
        return DeviceData(
            ePre: ePre, ePreLen: mf["e_pre_len"] as! Int, actEmbed: act, presets: presets,
            binCenters: farr("bin_centers"), q01: farr("norm_q01"), q99: farr("norm_q99"),
            mask: (mf["norm_mask"] as! [Any]).map { ($0 as! NSNumber).boolValue }, actLo: mf["act_lo"] as! Int)
    }

    // MARK: - Predict

    struct Result { let dof: [Float]; let tokens: [Int]; let visionMs: Double; let prefillMs: Double; let decodeMs: Double }

    /// image + preset instruction -> 7-DoF action.
    func predict(cgImage: CGImage, presetIndex: Int) async throws -> Result {
        guard let visionFn, let visionD, let llmFn, let llmD, let data else { throw Self.err("not loaded") }
        let preset = data.presets[presetIndex]

        // 1) vision -> img_embeds [256, HID]
        let tV = SuspendingClock.now
        let img = try await visionEncode(cgImage: cgImage, fn: visionFn, d: visionD)
        let visionMs = Self.ms(since: tV)

        // 2) host inputs_embeds = e_pre + img + e_post
        var seq = data.ePre                                   // [ePreLen*HID]
        seq.append(contentsOf: img)                           // [256*HID]
        seq.append(contentsOf: preset.ePost)                 // [len*HID]
        let S = data.ePreLen + N_IMG + preset.len

        // 3) KV state (persists across the S=1 loop)
        guard case .ndArray(let kd)? = llmD.stateDescriptor(of: "keyCache"),
              case .ndArray(let vd)? = llmD.stateDescriptor(of: "valueCache") else {
            throw Self.err("kv state descriptors missing")
        }
        let kvShape = [N_LAYERS, 1, N_KV, CAP, HEAD_DIM]
        var key = NDArray(descriptor: kd.resolvingDynamicDimensions(kvShape))
        var val = NDArray(descriptor: vd.resolvingDynamicDimensions(kvShape))
        fillNDArray(&key, as: Float16.self, with: [Float16](repeating: 0, count: kvShape.reduce(1,*)))
        fillNDArray(&val, as: Float16.self, with: [Float16](repeating: 0, count: kvShape.reduce(1,*)))

        // 4) prefill one position at a time
        print("[bitvla] vision \(Int(visionMs))ms; prefilling \(S) positions ...")
        let tP = SuspendingClock.now
        var logits = [Float](repeating: 0, count: data.binCenters.count + 1)
        for t in 0..<S {
            let row = Array(seq[(t*HID)..<((t+1)*HID)])
            logits = try await step(emb: row, pos: t, key: &key, val: &val, fn: llmFn, d: llmD)
            if t == 0 || (t + 1) % 50 == 0 {
                print(String(format: "[bitvla] prefill %d/%d (%.0fs)", t + 1, S, Self.ms(since: tP) / 1000))
            }
        }
        let prefillMs = Self.ms(since: tP)

        // 5) greedy 7 action tokens (argmax over 256 head rows), feed back via act_embed
        let tD = SuspendingClock.now
        var tokens: [Int] = []
        var pos = S
        for _ in 0..<ACT_DIM {
            let j = argmax(logits)
            tokens.append(data.actLo + j)
            let emb = Array(data.actEmbed[(j*HID)..<((j+1)*HID)])
            logits = try await step(emb: emb, pos: pos, key: &key, val: &val, fn: llmFn, d: llmD)
            pos += 1
        }
        let decodeMs = Self.ms(since: tD)

        let dof = detokenize(tokens: tokens, data: data)
        print(String(format: "[bitvla] vision %.0fms prefill %.0fms (%dtok) decode %.0fms -> %@",
                     visionMs, prefillMs, S, decodeMs, dof.map { String(format: "%.3f", $0) }.joined(separator: ", ")))
        return Result(dof: dof, tokens: tokens, visionMs: visionMs, prefillMs: prefillMs, decodeMs: decodeMs)
    }

    // One S=1 decode step: returns logits[256]. key/val mutate in place (the KV state).
    private func step(emb: [Float16], pos: Int, key: inout NDArray, val: inout NDArray,
                      fn: InferenceFunction, d: InferenceFunctionDescriptor) async throws -> [Float] {
        guard case .ndArray(let ed)? = d.inputDescriptor(of: "inputs_embeds"),
              case .ndArray(let pd)? = d.inputDescriptor(of: "position_ids"),
              case .ndArray(let od)? = d.outputDescriptor(of: "logits") else {
            throw Self.err("llm io descriptors missing")
        }
        var ie = NDArray(descriptor: ed.resolvingDynamicDimensions([1, 1, HID]))
        fillNDArray(&ie, as: Float16.self, with: emb)
        var pid = NDArray(descriptor: pd.resolvingDynamicDimensions([1, pos + 1]))
        fillNDArray(&pid, as: Int32.self, with: (0...Int32(pos)).map { $0 })
        var out = NDArray(descriptor: od.resolvingDynamicDimensions([1, 1, data!.binCenters.count + 1]))

        var states = InferenceFunction.MutableViews()
        states.insert(&key, for: "keyCache")
        states.insert(&val, for: "valueCache")
        var views = InferenceFunction.MutableViews()
        views.insert(&out, for: "logits")
        _ = try await fn.run(inputs: ["inputs_embeds": ie, "position_ids": pid],
                             states: consume states, outputViews: consume views)
        return flattenAsFloat(out)
    }

    // MARK: - Vision preprocess + encode

    private func visionEncode(cgImage: CGImage, fn: InferenceFunction,
                              d: InferenceFunctionDescriptor) async throws -> [Float16] {
        let pv = Self.preprocess(cgImage: cgImage)          // [3*224*224] CHW, normalized [-1,1]
        guard case .ndArray(let pin)? = d.inputDescriptor(of: "pixel_values"),
              case .ndArray(let e0)? = d.outputDescriptor(of: "img_embeds") else {
            throw Self.err("vision io descriptors missing")
        }
        var pArr = NDArray(descriptor: pin.resolvingDynamicDimensions([1, 3, 224, 224]))
        fillNDArray(&pArr, as: Float16.self, with: pv)
        var embOut = NDArray(descriptor: e0.resolvingDynamicDimensions([1, N_IMG, HID]))
        var views = InferenceFunction.MutableViews()
        views.insert(&embOut, for: "img_embeds")
        _ = try await fn.run(inputs: ["pixel_values": pArr],
                             states: InferenceFunction.MutableViews(), outputViews: consume views)
        let f = flattenAsFloat(embOut)
        return f.map { Float16($0) }
    }

    // SigLIP: resize 224x224, to [0,1], normalize mean/std 0.5 -> [-1,1]; CHW.
    private static func preprocess(cgImage: CGImage) -> [Float16] {
        let W = 224, H = 224
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))
        var out = [Float16](repeating: 0, count: 3 * H * W)
        for y in 0..<H {
            for x in 0..<W {
                let p = (y * W + x) * 4
                for c in 0..<3 {
                    let v = Float(rgba[p + c]) / 127.5 - 1.0    // [0,255] -> [-1,1]
                    out[c * H * W + y * W + x] = Float16(v)
                }
            }
        }
        return out
    }

    // MARK: - Action detokenize (OpenVLA 256-bin + BOUNDS_Q99)

    private func detokenize(tokens: [Int], data: DeviceData) -> [Float] {
        let total = data.actLo + data.binCenters.count + 1   // 128268
        var dof = [Float](repeating: 0, count: ACT_DIM)
        for i in 0..<min(ACT_DIM, tokens.count) {
            let disc = total - tokens[i]
            let bi = max(0, min(disc - 1, data.binCenters.count - 1))
            let norm = data.binCenters[bi]
            if i < data.mask.count, data.mask[i] {
                dof[i] = 0.5 * (norm + 1) * (data.q99[i] - data.q01[i] + 1e-8) + data.q01[i]
            } else { dof[i] = norm }
        }
        return dof
    }

    private func argmax(_ a: [Float]) -> Int {
        var bi = 0; var bv = -Float.greatestFiniteMagnitude
        for (i, v) in a.enumerated() where v > bv { bv = v; bi = i }
        return bi
    }

    private static func ms(since t: SuspendingClock.Instant) -> Double {
        let d = SuspendingClock.now - t; let (s, a) = d.components
        return (Double(s) + Double(a) / 1e18) * 1000
    }

    private static func err(_ m: String) -> Error {
        NSError(domain: "BitVLA", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
