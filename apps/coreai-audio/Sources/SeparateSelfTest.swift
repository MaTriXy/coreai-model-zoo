// Headless self-test for the Separate tab (SEPARATE_SELFTEST=1): load Mel-Band RoFormer
// through CoreAIKit's `KitSeparator`,
// separate the golden 8 s chunk, compare vs golden_vocals.f32 (cos + rms ratio — cos alone
// misses a global scale error), write a wav. Runs without the GUI (init()-launched).
import CoreAIKit
import Foundation

func runSeparateSelfTest() async {
    setvbuf(stdout, nil, _IONBF, 0)                    // unbuffer so logs surface live (GUI app)
    let logURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SEP_RESULT"] ?? "/tmp/sep_result.txt")
    try? "".write(to: logURL, atomically: true, encoding: .utf8)
    func log(_ s: String) {
        print("[SEP] \(s)")
        if let h = try? FileHandle(forWritingTo: logURL) { h.seekToEndOfFile(); h.write(Data(("[SEP] \(s)\n").utf8)); try? h.close() }
    }
    func finish(_ code: Int32) -> Never { log("EXIT \(code)"); exit(code) }
    guard let root = SeparateAssets.root, let murl = SeparateAssets.modelURL else {
        log("FAIL: assets not found at \(SeparateAssets.location.path)"); finish(2)
    }
    func readF32(_ name: String) -> [Float]? {
        guard let d = try? Data(contentsOf: root.appendingPathComponent(name)) else { return nil }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    guard let rawFlat = readF32("golden_raw.f32"), let goldFlat = readF32("golden_vocals.f32") else {
        log("FAIL: golden_raw/golden_vocals.f32 missing"); finish(2)
    }
    let C = KitSeparator.chunkSamples
    let mix = [Array(rawFlat[0..<C]), Array(rawFlat[C..<2 * C])]
    do {
        let t0 = ContinuousClock().now
        let sep = try await KitSeparator(bundleAt: murl, computeUnits: .gpu)
        log("loaded in \(ms(since: t0))")
        let g0 = ContinuousClock().now
        let voc = try await sep.separateChunk(mix)               // single-chunk path (matches golden)
        let dt = msVal(since: g0) / 1000
        log(String(format: "separate 8s chunk in %.2fs (%.1f× real-time)", dt, Double(C) / Double(KitSeparator.sampleRate) / dt))

        // cos + rms ratio vs golden (ch0 then ch1)
        let out = voc[0] + voc[1]
        var dot = 0.0, na = 0.0, nb = 0.0
        for k in 0..<out.count { let a = Double(out[k]), b = Double(goldFlat[k]); dot += a * b; na += a * a; nb += b * b }
        let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
        let rmsRatio = (na / nb).squareRoot()
        log(String(format: "vs golden: cos=%.6f  rms_ratio=%.4f", cos, rmsRatio))
        let pass = cos >= 0.99 && abs(rmsRatio - 1) < 0.05
        writeStereoWav(voc, sr: KitSeparator.sampleRate,
                       to: URL(fileURLWithPath: ProcessInfo.processInfo.environment["SEP_OUT"] ?? "/tmp/separate_selftest.wav"))
        log(pass ? "PASS" : "CHECK (cos or rms off — likely a host DSP mismatch)")
        finish(pass ? 0 : 3)
    } catch { log("FAIL: \(error)"); finish(4) }
}

private func ms(since t: ContinuousClock.Instant) -> String { String(format: "%.2fs", msVal(since: t) / 1000) }
private func msVal(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock().now - t
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

private func writeStereoWav(_ s: [[Float]], sr: Int, to url: URL) {
    let n = s[0].count, ch = 2, bytes = 2, dataLen = n * ch * bytes
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(UInt16(ch)); u32(UInt32(sr))
    u32(UInt32(sr * ch * bytes)); u16(UInt16(ch * bytes)); u16(16)
    d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
    for k in 0..<n { for c in 0..<ch { let x = max(-1, min(1, s[c][k])); u16(UInt16(bitPattern: Int16(x * 32767))) } }
    try? d.write(to: url)
}
