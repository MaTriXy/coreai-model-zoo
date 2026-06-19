// ContentView — on-device audio understanding: record from the mic, choose a file, or use the
// demo clip, then ask "what do you hear?". The local Qwen2.5-Omni Thinker describes the SOUNDS
// (events / texture / emotion), not a transcript.

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var backend = AudioBackend()
    @StateObject private var recorder = AudioRecorder()
    @State private var question = "What do you hear?"
    @State private var clipName = "No audio loaded."
    @State private var showImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("coreai-audio — on-device audio understanding")
                .font(.title2).bold()
            Text("Qwen2.5-Omni Thinker on Core AI. Describes the *sounds* it hears — not a transcript.")
                .font(.callout).foregroundStyle(.secondary)

            if !backend.loaded {
                Button { Task { await backend.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(backend.busy)
            }

            HStack(spacing: 12) {
                Button {
                    recorder.isRecording ? stopRecording() : startRecording()
                } label: {
                    Label(recorder.isRecording ? "Stop" : "Record",
                          systemImage: recorder.isRecording ? "stop.circle.fill" : "mic.circle")
                }.disabled(!backend.loaded || backend.busy)
                    .tint(recorder.isRecording ? .red : .accentColor)

                Button { showImporter = true } label: {
                    Label("Choose…", systemImage: "waveform")
                }.disabled(!backend.loaded || backend.busy)

                Button { useDemo() } label: {
                    Label("Demo noise", systemImage: "speaker.wave.2")
                }.disabled(!backend.loaded || backend.busy)
            }

            Text(clipName).font(.footnote).foregroundStyle(.secondary)

            HStack {
                TextField("Question", text: $question).textFieldStyle(.roundedBorder)
                Button("Ask") { Task { await backend.ask(question) } }
                    .disabled(!backend.loaded || backend.busy)
            }

            if backend.busy { ProgressView().controlSize(.small) }

            ScrollView {
                Text(backend.answer.isEmpty ? " " : backend.answer)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }.frame(minHeight: 110)

            Text(backend.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)
        #endif
        .fileImporter(
            isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first { loadFile(url) }
        }
        .task {
            if CommandLine.arguments.contains("--selftest") {
                await backend.load()
                useDemo()
                await backend.ask("What do you hear?")
            }
        }
    }

    private func startRecording() {
        do { try recorder.start(); clipName = "Recording…" } catch {
            clipName = "Mic error: \(error.localizedDescription)"
        }
    }
    private func stopRecording() {
        let pcm = recorder.stop()
        clipName = String(format: "Mic clip (%.1fs)", Double(pcm.count) / 16000)
        Task { try? await backend.attach(samples: pcm) }
    }
    private func useDemo() {
        let pcm = AudioRecorder.demoNoise()
        clipName = "Demo: white noise (4s)"
        Task { try? await backend.attach(samples: pcm) }
    }
    private func loadFile(_ url: URL) {
        guard let pcm = AudioRecorder.load16kMono(url) else {
            clipName = "Could not decode \(url.lastPathComponent)."; return
        }
        clipName = "\(url.lastPathComponent) (\(String(format: "%.1f", Double(pcm.count) / 16000))s)"
        Task { try? await backend.attach(samples: pcm) }
    }
}
