// KokoroView — the "Speak" tab: Kokoro-82M text-to-speech on Core AI. Demo phrases
// are phonemized ahead of time (G2P is host-side; this build carries no MLX/espeak
// dependency); the on-device pipeline (predictor/prosody/vocoder + Swift host DSP)
// runs the rest.
import CoreAIKit
import SwiftUI

struct DemoPhrase: Codable, Hashable {
    let text: String
    let ids: [Int]
}

enum KokoroAssets {
    static let repo = "mlboydaisuke/Kokoro-82M-CoreAI"
    static let bundleNames = ["kokoro_predictor.aimodel", "kokoro_prosody.aimodel", "kokoro_vocoder.aimodel"]

    /// Small files (vocab/voices/…) bundled in the app; dev fallback to the source tree.
    static var resources: URL {
        Bundle.main.url(forResource: "KokoroResources", withExtension: nil)
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("KokoroResources")
    }

    /// A `.aimodel` bundle: local dev copy if present, else the cached/downloaded HF copy.
    static func bundle(_ name: String, store: ModelStore,
                       progress: @escaping @Sendable (DownloadProgress) -> Void) async throws -> URL {
        let dev = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("KokoroAssets").appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: dev.path) { return dev }
        let id = ModelID(repo, path: name)
        if let u = store.localURL(for: id) { return u }
        return try await store.download(id, progress: progress)
    }
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
    private let store = ModelStore()

    func load() async {
        busy = true; status = "Loading…"
        do {
            let res = KokoroAssets.resources
            // The 3 .aimodel bundles: dev copy if present, else download from HF (cached).
            var urls: [URL] = []
            for name in KokoroAssets.bundleNames {
                let u = try await KokoroAssets.bundle(name, store: store) { p in
                    Task { @MainActor in self.status = "Downloading \(name) — \(Int(p.fraction * 100))%" }
                }
                urls.append(u)
            }
            let t = try await KokoroTTS(
                predictor: urls[0], prosody: urls[1], vocoder: urls[2],
                vocab: res.appendingPathComponent("vocab.json"),
                lLinear: res.appendingPathComponent("l_linear.bin"))
            let vdir = res.appendingPathComponent("voices")
            for f in (try? FileManager.default.contentsOfDirectory(at: vdir, includingPropertiesForKeys: nil)) ?? []
            where f.pathExtension == "bin" {
                await t.loadVoice(f.deletingPathExtension().lastPathComponent, url: f)
            }
            let data = try Data(contentsOf: res.appendingPathComponent("demo_phrases.json"))
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
