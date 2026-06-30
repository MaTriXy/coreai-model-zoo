// MusicGenView — the "Music" tab: text -> ~11s of music/audio on-device (Stable Audio Open Small,
// zoo's first generative audio). Drives MusicGenModel (3 Core AI bundles + host sampler).
import SwiftUI

struct MusicGenView: View {
    @StateObject private var model = MusicGenModel()
    @State private var prompt = "128 BPM tech house drum loop"
    @State private var seconds: Double = 11

    private let examples = [
        "128 BPM tech house drum loop",
        "warm lo-fi hip hop beat with vinyl crackle",
        "epic orchestral trailer, brass and drums",
        "ambient pad, slow and dreamy",
        "funky disco bassline, 120 BPM",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Music — on-device generation").font(.title2).bold()
            Text("Stable Audio Open Small on Core AI. Describe a sound; get ~11 s of 44.1 kHz stereo, generated entirely on device.")
                .font(.callout).foregroundStyle(.secondary)

            if !model.loaded {
                Button { Task { await model.load() } } label: { Label("Load model", systemImage: "arrow.down.circle") }
                    .disabled(model.busy)
            }

            TextField("Describe the music…", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder).lineLimit(1...3)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack { ForEach(examples, id: \.self) { e in
                    Button(e) { prompt = e }.buttonStyle(.bordered).controlSize(.small).lineLimit(1)
                } }
            }

            HStack {
                Text("Length: \(Int(seconds)) s").font(.callout).monospacedDigit()
                Slider(value: $seconds, in: 1...11, step: 1)
            }

            HStack(spacing: 12) {
                Button { Task { await model.generate(prompt: prompt, seconds: Float(seconds)) } } label: {
                    Label("Generate", systemImage: "music.note")
                }.disabled(!model.loaded || model.busy).buttonStyle(.borderedProminent)
                Button { model.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                    .disabled(!model.loaded)
                #if os(macOS)
                Button { saveWav() } label: { Label("Save…", systemImage: "square.and.arrow.down") }
                    .disabled(model.lastStats.isEmpty || model.busy)
                #endif
            }

            if model.busy { ProgressView().controlSize(.small) }
            if !model.lastStats.isEmpty {
                Label(model.lastStats, systemImage: "bolt.fill").font(.footnote).foregroundStyle(.secondary)
            }
            Text(model.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)
        #endif
    }

    #if os(macOS)
    private func saveWav() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "music.wav"; panel.allowedContentTypes = [.wav]
        if panel.runModal() == .OK, let url = panel.url { model.saveWav(to: url) }
    }
    #endif
}
