// Headless self-test for the Dialogue tab (DIALOGUE_SELFTEST=1): load VibeVoice through
// CoreAIKit's `KitDialogue` from the sideloaded assets, perform a two-speaker script, and report
// load time, turns, voices and real-time factor. This is the *kit* path — the app-local
// VibeVoiceSelfTest exercises the raw Core AI runtime instead, so running both tells you whether
// a regression is in the graphs or in the kit host code.
import CoreAIKit
import Foundation

func runDialogueSelfTest() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let logURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DLG_RESULT"] ?? "/tmp/dlg_result.txt")
    try? "".write(to: logURL, atomically: true, encoding: .utf8)
    func log(_ s: String) {
        print("[DLG] \(s)")
        if let h = try? FileHandle(forWritingTo: logURL) {
            h.seekToEndOfFile(); h.write(Data(("[DLG] \(s)\n").utf8)); try? h.close()
        }
    }
    func finish(_ code: Int32) -> Never { log("EXIT \(code)"); exit(code) }

    guard let paths = DialogueAssets.paths() else {
        log("FAIL: host assets not found at \(DialogueAssets.location.path) (need glue/ voices/ embed/)")
        finish(2)
    }
    do {
        let t0 = Date()
        let dialogue = try await KitDialogue(paths: paths, computeUnits: .gpu)
        log(String(format: "loaded %d voices in %.1fs", dialogue.voices.count, Date().timeIntervalSince(t0)))

        let g0 = Date()
        let (audio, turns) = try await dialogue.perform(DialogueModel.sampleScript)
        let dt = Date().timeIntervalSince(g0)
        log("turns: " + turns.map { "S\($0.speaker)=\($0.voice)" }.joined(separator: ", "))
        log(String(format: "%.2fs audio in %.1fs (%.2fx real-time), %d samples @ %d Hz",
                   audio.seconds, dt, audio.seconds / max(dt, 0.001), audio.samples.count, audio.sampleRate))

        // sanity: two distinct voices, audible output, no NaNs
        let peak = audio.samples.map { abs($0) }.max() ?? 0
        let finite = audio.samples.allSatisfy { $0.isFinite }
        let distinct = Set(turns.map(\.voice)).count
        let pass = turns.count == 2 && distinct == 2 && finite && peak > 0.01
        if let out = ProcessInfo.processInfo.environment["DLG_OUT"] {
            writeMonoWav(audio.samples, sr: audio.sampleRate, to: URL(fileURLWithPath: out))
        }
        log(String(format: "peak %.3f, finite %@, distinct voices %d -> %@",
                   peak, finite ? "yes" : "NO", distinct, pass ? "PASS" : "CHECK"))
        finish(pass ? 0 : 3)
    } catch {
        log("FAIL: \(error)")
        finish(4)
    }
}

private func writeMonoWav(_ s: [Float], sr: Int, to url: URL) {
    let n = s.count, bytes = 2, dataLen = n * bytes
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(1); u32(UInt32(sr))
    u32(UInt32(sr * bytes)); u16(UInt16(bytes)); u16(16)
    d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
    for v in s { u16(UInt16(bitPattern: Int16(max(-1, min(1, v)) * 32767))) }
    try? d.write(to: url)
}
