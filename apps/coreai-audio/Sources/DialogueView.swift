// DialogueView — the "Dialogue" tab: write a two-speaker script, hear the conversation
// (VibeVoice-Realtime-0.5B, the zoo's first multi-speaker TTS). Each speaker gets its own voice
// preset. Generate a conversation here, then take it to Transcribe → Diarize and watch the zoo
// label who spoke when.
import SwiftUI

struct DialogueView: View {
    @StateObject private var model = DialogueModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Dialogue — multi-speaker").font(.title2).bold()
            Text("VibeVoice on Core AI. One voice preset per speaker, generated on device.")
                .font(.callout).foregroundStyle(.secondary)

            if !model.loaded {
                Button { Task { await model.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }
                .disabled(model.busy)
            }

            TextEditor(text: $model.script)
                .font(.body.monospaced())
                .frame(minHeight: 110)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                .disabled(model.busy)

            if model.loaded {
                HStack(spacing: 12) {
                    voicePicker("Speaker 1", selection: $model.voice1)
                    voicePicker("Speaker 2", selection: $model.voice2)
                }
            }

            HStack(spacing: 12) {
                Button { Task { await model.generate() } } label: {
                    Label("Generate", systemImage: "person.2.wave.2")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.loaded || model.busy)

                if model.haveAudio {
                    Button { model.play() } label: { Label("Play", systemImage: "play.fill") }
                    Button { model.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                    #if os(macOS)
                    Button { save() } label: { Image(systemName: "square.and.arrow.down") }
                    #endif
                }
            }

            if model.busy { ProgressView().progressViewStyle(.linear) }
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
    }

    private func voicePicker(_ label: String, selection: Binding<String>) -> some View {
        Picker(label, selection: selection) {
            ForEach(model.voices, id: \.self) { Text($0).tag($0) }
        }
        .pickerStyle(.menu)
        .disabled(model.busy)
    }

    #if os(macOS)
    private func save() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "dialogue.wav"
        panel.allowedContentTypes = [.wav]
        if panel.runModal() == .OK, let url = panel.url { model.saveWav(to: url) }
    }
    #endif
}
