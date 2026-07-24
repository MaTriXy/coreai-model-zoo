// DialogueModel — the "Dialogue" tab VM. Drives CoreAIKit's `KitDialogue`
// (VibeVoice-Realtime-0.5B, the zoo's first multi-speaker TTS): write a "Speaker 1 / Speaker 2"
// script, get the conversation back, one voice preset per speaker.
//
// This is the same code path an app gets from `KitDialogue(catalog: "vibevoice-realtime-0.5b")`,
// pointed at the sideloaded assets instead of the download cache. Pairs with the Transcribe tab's
// diarizer: generate a conversation here, then have Sortformer say who spoke when.
//
// Assets (`VibeVoiceAssets` root): the five graph bundles + `coreai_host`'s glue/ voices/ embed/,
// staged by conversion/vibevoice/sideload_ios.sh (device) or read from the conversion tree (Mac).
import CoreAIKit
import CoreAIKitVision
import Foundation
import SwiftUI

enum DialogueAssets {
    /// Where the sideloaded graphs + host assets live.
    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("VibeVoiceAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VibeVoiceAssets")
        #endif
    }

    /// On Mac the graphs are the un-compiled bundles in the conversion tree (one dir per graph);
    /// on device they are the flat AOT `.aimodelc` the sideload script pushes.
    static func paths() -> VibeVoicePaths? {
        let root = location
        let glue = root.appendingPathComponent("glue")
        let voices = root.appendingPathComponent("voices")
        let embed = root.appendingPathComponent("embed/embed_tokens_fp16.bin")
        let fm = FileManager.default
        guard fm.fileExists(atPath: glue.appendingPathComponent("glue.json").path),
              fm.fileExists(atPath: voices.appendingPathComponent("index.json").path),
              fm.fileExists(atPath: embed.path)
        else { return nil }

        #if os(macOS)
        // conversion/vibevoice/artifacts/<name>/<name>.aimodel
        let art = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("conversion/vibevoice/artifacts")
        func u(_ n: String) -> URL { art.appendingPathComponent("\(n)/\(n).aimodel") }
        guard fm.fileExists(atPath: u("vibevoice_mainlm_fp16_decode_cl512").path) else { return nil }
        return VibeVoicePaths(
            mainLM: u("vibevoice_mainlm_fp16_decode_cl512"),
            ttsLM: u("vibevoice_ttslm_fp16_decode_cl512"),
            head: u("vibevoice_diffusion_head_fp16"),
            connector: u("vibevoice_connector_fp16"),
            decoder: u("vibevoice_decoder_fp16_t64"),
            glueDir: glue, voicesDir: voices, embedTokens: embed)
        #else
        return VibeVoicePaths.inBundleDir(root, glueDir: glue, voicesDir: voices, embedTokens: embed)
        #endif
    }
}

@MainActor
final class DialogueModel: ObservableObject {
    static let sampleScript = """
        Speaker 1: Did you know this runs entirely on the phone?
        Speaker 2: No cloud at all? That is wild.
        """

    @Published var script = DialogueModel.sampleScript
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var stats = ""
    @Published var haveAudio = false
    @Published var voices: [String] = []
    @Published var voice1 = ""
    @Published var voice2 = ""

    private var dialogue: KitDialogue?
    private let player = AudioPlayer()
    private var samples: [Float] = []
    private let sr = 24_000.0

    func load() async {
        busy = true; defer { busy = false }
        guard let paths = DialogueAssets.paths() else {
            status = "Assets not found — stage them at \(DialogueAssets.location.path)"
            return
        }
        status = "Loading VibeVoice (5 graphs)…"
        do {
            let clock = ContinuousClock(); let t0 = clock.now
            let d = try await KitDialogue(paths: paths, computeUnits: .gpu)
            dialogue = d
            voices = d.voices.map(\.name)
            let preferred = d.defaultVoices
            voice1 = preferred.first ?? voices.first ?? ""
            voice2 = preferred.dropFirst().first ?? voice1
            loaded = true
            status = String(format: "Ready — %d voices, loaded in %.1f s.",
                            voices.count, seconds(since: t0, clock))
        } catch { status = "Load failed: \(error)" }
    }

    func generate() async {
        guard let dialogue, !busy else { return }
        busy = true; haveAudio = false; defer { busy = false }
        status = "Generating…"
        do {
            let clock = ContinuousClock(); let t0 = clock.now
            let (audio, turns) = try await dialogue.perform(script, voices: [voice1, voice2])
            let el = seconds(since: t0, clock)
            samples = audio.samples
            haveAudio = !samples.isEmpty
            stats = String(format: "%d turns · %.1f s audio in %.1f s (%.2f× real-time)",
                           turns.count, audio.seconds, el, audio.seconds / max(el, 0.001))
            status = "Done — \(stats)"
            play()
        } catch { status = "Generation failed: \(error)" }
    }

    func play() {
        guard haveAudio else { return }
        player.reset(sampleRate: sr)
        player.play(samples, sampleRate: sr)
    }

    func stop() { player.reset(sampleRate: sr) }

    func saveWav(to url: URL) {
        guard haveAudio else { return }
        let n = samples.count, bytes = 2, dataLen = n * bytes
        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataLen)); d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(1); u32(UInt32(sr))
        u32(UInt32(sr) * UInt32(bytes)); u16(UInt16(bytes)); u16(16)
        d.append("data".data(using: .ascii)!); u32(UInt32(dataLen))
        for v in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, v)) * 32767))) }
        try? d.write(to: url)
    }

    private func seconds(since t: ContinuousClock.Instant, _ clock: ContinuousClock) -> Double {
        let d = clock.now - t
        return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}
