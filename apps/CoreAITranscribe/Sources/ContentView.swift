import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var engine = TranscribeEngine()
    @StateObject private var recorder = Recorder()
    @State private var selectedModel = TranscribeEngine.catalog.first
    @State private var showingAudioImporter = false

    var body: some View {
        content
            .fileImporter(isPresented: $showingAudioImporter, allowedContentTypes: [.audio]) { result in
                if case .success(let url) = result { engine.transcribe(audioURL: url) }
            }
    }

    @ViewBuilder private var content: some View {
        #if os(macOS)
        VStack(alignment: .leading, spacing: 16) { controls; transcriptView }.padding(20)
        #else
        NavigationStack {
            ScrollView { VStack(alignment: .leading, spacing: 16) { controls; transcriptView }.padding(16) }
                .navigationTitle("CoreAI Transcribe")
                .navigationBarTitleDisplayMode(.inline)
        }
        #endif
    }

    @ViewBuilder private var controls: some View {
        groupBox("Model") {
            HStack {
                if let m = selectedModel {
                    Button { engine.loadFromHub(m) } label: {
                        Label("Download & Load", systemImage: "arrow.down.circle")
                    }.disabled(engine.status.isBusy)
                }
            }
            statusLine
            if case .downloading = engine.status { DownloadBar(downloader: engine.downloader) }
        }

        groupBox("Audio") {
            HStack(spacing: 12) {
                Button { showingAudioImporter = true } label: {
                    Label("Choose File…", systemImage: "waveform")
                }.disabled(!engine.canTranscribe)

                Button {
                    if recorder.isRecording {
                        if let url = recorder.stop() { engine.transcribe(audioURL: url) }
                    } else {
                        recorder.requestPermissionAndStart()
                    }
                } label: {
                    Label(recorder.isRecording ? "Stop & Transcribe" : "Record",
                          systemImage: recorder.isRecording ? "stop.circle.fill" : "mic.circle")
                }
                .tint(recorder.isRecording ? .red : nil)
                .disabled(!engine.canTranscribe && !recorder.isRecording)
            }
            Text("Transcribes one 30 s window on-device.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var transcriptView: some View {
        groupBox("Transcript") {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 10).fill(Color.gray.opacity(0.12))
                if engine.status == .transcribing {
                    ProgressView().padding()
                } else if engine.transcript.isEmpty {
                    Text("Pick or record audio, then it appears here.")
                        .foregroundStyle(.secondary).padding(12)
                } else {
                    Text(engine.transcript).textSelection(.enabled).padding(12)
                }
            }
            .frame(minHeight: 160, alignment: .topLeading)
            if let s = engine.seconds, !engine.transcript.isEmpty {
                Text(String(format: "%.2fs", s)).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var statusLine: some View {
        HStack(spacing: 8) {
            if engine.status.isBusy { ProgressView().controlSize(.small) }
            Text(engine.status.label).font(.caption)
                .foregroundStyle({ if case .error = engine.status { return Color.red }
                                   if case .ready = engine.status { return Color.green }
                                   return Color.secondary }())
                .lineLimit(2)
            Spacer()
        }
    }

    @ViewBuilder
    private func groupBox<C: View>(_ title: String, @ViewBuilder _ c: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            c()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct DownloadBar: View {
    @ObservedObject var downloader: ModelDownloader
    var body: some View {
        VStack(spacing: 6) {
            ProgressView(value: downloader.fraction)
            Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}
