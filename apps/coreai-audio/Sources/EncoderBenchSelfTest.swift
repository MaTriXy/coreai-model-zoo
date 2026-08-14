// Headless encoder bench: ENCBENCH_SELFTEST=1 times one stateless graph on one compute unit.
//
// Built for the question PR #13 asks — is an ANE-authored FastConformer encoder faster than the
// GPU-authored one we ship? — but it is model-agnostic: any single-input, single-output
// `.aimodelc` under Documents/ can be timed by pointing ENCBENCH_BUNDLE at it. Load, first call
// (specialization) and warm median are reported separately, because for a 1 GB graph they answer
// different questions.
//
//   ENCBENCH_SELFTEST=1
//   ENCBENCH_BUNDLE=Models/EncBench/gpu_track.h18p.aimodelc   (relative to Documents/)
//   ENCBENCH_UNIT=gpu|ane|cpu                                  (default gpu)
//   ENCBENCH_PASSES=20
//   ENCBENCH_TAG=whatever                                      (echoed into the result line)
//
// Writes Documents/encbench_result.txt (pullable via devicectl) + NSLog.

import CoreAIKitVision
import Foundation

func runEncoderBenchSelfTest() async {
    NSLog("EncBench: start")
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let out = docs.appending(path: "encbench_result.txt")

    func write(_ s: String) {
        try? s.write(to: out, atomically: true, encoding: .utf8)
        NSLog("EncBench: %@", s)
    }

    guard let rel = env["ENCBENCH_BUNDLE"] else {
        write("FAIL: ENCBENCH_BUNDLE not set")
        return
    }
    let url = docs.appending(path: rel)
    guard fm.fileExists(atPath: url.path) else {
        write("FAIL: no bundle at \(url.path)")
        return
    }
    let tag = env["ENCBENCH_TAG"] ?? rel
    let unitName = (env["ENCBENCH_UNIT"] ?? "gpu").lowercased()
    let units: GraphModel.ComputeUnits = switch unitName {
    case "ane", "neuralengine", "neural-engine": .neuralEngine
    case "cpu": .cpu
    default: .gpu
    }
    let passes = Int(env["ENCBENCH_PASSES"] ?? "") ?? 20

    do {
        let t0 = Date()
        let model = try await GraphModel(contentsOf: url, computeUnits: units)
        let loadMs = Date().timeIntervalSince(t0) * 1000
        guard let inName = model.inputNames.first,
              let outName = model.outputNames.first,
              let shape = model.shape(ofInput: inName)
        else {
            write("FAIL: could not read the graph's input/output descriptors")
            return
        }
        let count = shape.reduce(1, *)
        NSLog("EncBench: loaded (%.0f ms), input %@ %@", loadMs, inName, "\(shape)")

        // Deterministic pseudo-random input. fp16 timing is not value-dependent in any way that
        // matters here, but a constant would let the compiler fold work away, and zeros can hit
        // denormal paths — so: a fixed LCG, identical for every bundle and unit.
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        var scalars = [Float16]()
        scalars.reserveCapacity(count)
        for _ in 0..<count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float(state >> 40) / Float(1 << 24)          // [0,1)
            scalars.append(Float16((u - 0.5) * 1.6))
        }
        let input: [String: TensorValue] = [inName: .float16(scalars, shape: shape)]

        let t1 = Date()
        _ = try await model.run(input)
        let firstMs = Date().timeIntervalSince(t1) * 1000

        var ms = [Double]()
        for _ in 0..<passes {
            let t = Date()
            _ = try await model.run(input)
            ms.append(Date().timeIntervalSince(t) * 1000)
        }
        let sorted = ms.sorted()
        let median = sorted[sorted.count / 2]
        let line = String(
            format: "OK tag=%@ unit=%@ passes=%d load=%.0fms first=%.1fms "
                  + "median=%.2fms best=%.2fms worst=%.2fms out=%@",
            tag, unitName, passes, loadMs, firstMs, median, sorted.first ?? 0,
            sorted.last ?? 0, outName)
        write(line)
    } catch {
        write("FAIL: \(error)")
    }
}
