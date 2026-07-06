// SeparateView — the "Separate" tab: split a song into vocals + instrumental on-device
// (Mel-Band RoFormer, zoo's first source separation). Pick a file or use the demo clip,
// then play either stem. Pairs with the Music tab (generate a track, then rip its stems).
import SwiftUI
import UniformTypeIdentifiers

struct SeparateView: View {
    @StateObject private var model = SeparateModel()
    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Separate — vocals / instrumental").font(.title2).bold()
            Text("Mel-Band RoFormer on Core AI. Split any song into an acapella and a karaoke track, entirely on device.")
                .font(.callout).foregroundStyle(.secondary)

            if !model.loaded {
                Button { Task { await model.load() } } label: { Label("Load model", systemImage: "arrow.down.circle") }
                    .disabled(model.busy)
            }

            HStack(spacing: 12) {
                Button { importing = true } label: { Label("Choose song…", systemImage: "music.note.list") }
                    .disabled(!model.loaded || model.busy)
                Button { Task { await model.separate(nil) } } label: { Label("Demo clip", systemImage: "waveform") }
                    .disabled(!model.loaded || model.busy)
            }

            if model.busy {
                ProgressView(value: model.progress).progressViewStyle(.linear)
            }

            if model.haveResult {
                HStack(spacing: 12) {
                    Button { model.playVocals() } label: { Label("Vocals", systemImage: "music.mic") }
                        .buttonStyle(.borderedProminent)
                    Button { model.playInstrumental() } label: { Label("Instrumental", systemImage: "guitars") }
                        .buttonStyle(.bordered)
                    Button { model.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                    #if os(macOS)
                    Button { save(true) } label: { Image(systemName: "square.and.arrow.down") }
                    #endif
                }
            }

            if !model.stats.isEmpty {
                Label(model.stats, systemImage: "bolt.fill").font(.footnote).foregroundStyle(.secondary)
            }
            Text(model.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)
        #endif
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio]) { result in
            guard case .success(let url) = result else { return }
            Task {
                guard let mix = AudioLoader.load44kStereo(url) else { model.status = "Could not read that file."; return }
                await model.separate(mix)
            }
        }
    }

    #if os(macOS)
    private func save(_ vocals: Bool) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = vocals ? "vocals.wav" : "instrumental.wav"
        panel.allowedContentTypes = [.wav]
        if panel.runModal() == .OK, let url = panel.url { model.saveWav(vocals, to: url) }
    }
    #endif
}
