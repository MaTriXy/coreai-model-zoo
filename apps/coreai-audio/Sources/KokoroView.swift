// KokoroView — the "Speak" tab: Kokoro-82M text-to-speech on Core AI. Demo phrases
// are phonemized ahead of time (G2P is host-side; this build carries no MLX/espeak
// dependency); the on-device pipeline (predictor/prosody/vocoder + Swift host DSP)
// runs the rest.
import SwiftUI

struct DemoPhrase: Codable, Hashable {
    let text: String
    let ids: [Int]
}

@MainActor
final class KokoroVM: ObservableObject {
    @Published var status = "Tap Load to start."
    @Published var loaded = false
    @Published var busy = false
    @Published var voices: [String] = []
    @Published var phrases: [DemoPhrase] = []

    private var tts: KokoroTTS?
    private let player = AudioPlayer()

    // The assets (3 .aimodel bundles + voices/ + vocab.json + l_linear.bin +
    // demo_phrases.json). On device they live in the app bundle / Documents (the
    // small files ship in the app; the .aimodel download from the HF repo). On a
    // dev build we fall back to the source tree (resolved relative to this file).
    private var assets: URL {
        if let r = Bundle.main.url(forResource: "KokoroAssets", withExtension: nil) { return r }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("KokoroAssets")
        if FileManager.default.fileExists(atPath: docs.path) { return docs }
        return URL(fileURLWithPath: #filePath)          // …/Sources/KokoroView.swift
            .deletingLastPathComponent().appendingPathComponent("KokoroAssets")
    }

    func load() async {
        busy = true; status = "Loading bundles (CPU)…"
        do {
            let t = try await KokoroTTS(assets: assets)
            let vdir = assets.appendingPathComponent("voices")
            for f in (try? FileManager.default.contentsOfDirectory(at: vdir, includingPropertiesForKeys: nil)) ?? []
            where f.pathExtension == "bin" {
                await t.loadVoice(f.deletingPathExtension().lastPathComponent, url: f)
            }
            let data = try Data(contentsOf: assets.appendingPathComponent("demo_phrases.json"))
            phrases = try JSONDecoder().decode([DemoPhrase].self, from: data)
            voices = await t.availableVoices()
            tts = t; loaded = true
            status = "Ready — \(voices.count) voices."
        } catch {
            status = "Load failed: \(error)"
        }
        busy = false
    }

    func speak(_ p: DemoPhrase, voice: String) async {
        guard let tts else { return }
        busy = true; status = "Synthesizing “\(p.text.prefix(28))”…"
        do {
            let t0 = Date()
            let audio = try await tts.synthesize(ids: p.ids, voice: voice)
            let ms = Date().timeIntervalSince(t0) * 1000
            status = String(format: "%.2f s audio in %.0f ms (%@)", Double(audio.count) / 24000, ms, voice)
            player.play(audio)
        } catch {
            status = "Error: \(error)"
        }
        busy = false
    }
}

struct KokoroView: View {
    @StateObject private var vm = KokoroVM()
    @State private var voice = "af_heart"

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Kokoro-82M — on-device text-to-speech")
                .font(.title2).bold()
            Text("StyleTTS2 + iSTFTNet on Core AI (CPU). Three bundles + host DSP, no network.")
                .font(.callout).foregroundStyle(.secondary)

            if !vm.loaded {
                Button { Task { await vm.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(vm.busy)
            } else {
                Picker("Voice", selection: $voice) {
                    ForEach(vm.voices, id: \.self) { Text($0).tag($0) }
                }.pickerStyle(.menu)

                ForEach(vm.phrases, id: \.self) { p in
                    Button {
                        Task { await vm.speak(p, voice: voice) }
                    } label: {
                        Label(p.text, systemImage: "speaker.wave.2.fill")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .disabled(vm.busy)
                }
            }

            if vm.busy { ProgressView().controlSize(.small) }
            Text(vm.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)
        #endif
    }
}
