import SwiftUI
import AVKit

/// AppKit AVPlayerView wrapper. SwiftUI's `VideoPlayer` (_AVKit_SwiftUI) can hit a Swift
/// metadata fatalError on macOS; AVPlayerView is the robust path.
struct MacVideoPlayer: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.player = player
        v.controlsStyle = .inline
        v.videoGravity = .resizeAspect
        return v
    }
    func updateNSView(_ v: AVPlayerView, context: Context) {
        if v.player !== player { v.player = player }
    }
}

enum Config {
    // Directory holding app_backend.py. Bundles + ckpts + ltx_video live under `runtime`.
    static let defaultBackendDir = "~/Code/coreai/coreai-models-community/apps/CoreAIVideo"
    static let defaultRuntime    = "~/CoreAIVideoRuntime"
    static let defaultCoreAI     = "~/Code/coreai"
    static let defaultPython     = "~/Code/coreai/coreai-models/.venv/bin/python"
}

struct ContentView: View {
    @StateObject private var gen = Generator()
    @State private var prompt =
        "A clear glass of water on a wooden table, slow motion droplet falling into it creating ripples, cinematic, soft natural light"
    @State private var seed = 42
    @State private var player: AVPlayer?
    @State private var showPaths = false
    @AppStorage("python") private var python = Config.defaultPython
    @AppStorage("backendDir") private var backendDir = Config.defaultBackendDir
    @AppStorage("runtime") private var runtime = Config.defaultRuntime
    @AppStorage("coreai") private var coreai = Config.defaultCoreAI

    var body: some View {
        HSplitView {
            controls
                .frame(minWidth: 340, idealWidth: 360, maxWidth: 460)
                .padding()
            viewer
                .frame(minWidth: 540, minHeight: 540)
        }
        .onAppear { gen.start(python: python, backendDir: backendDir, runtime: runtime, coreai: coreai) }
        .onChange(of: gen.resultURL) { _, url in loadVideo(url) }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("CoreAIVideo · LTX-Video 2B").font(.title2).bold()
            Text("Text → video, all three nets (T5 + DiT + video VAE) on Apple Core AI. 512×768 · 49 frames · 8 steps.")
                .font(.caption).foregroundStyle(.secondary)

            GroupBox("Prompt") {
                TextEditor(text: $prompt)
                    .font(.body).frame(height: 120)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
            }

            GroupBox("Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Seed")
                        TextField("seed", value: $seed, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 100)
                        Button { seed = Int.random(in: 0...999_999) } label: {
                            Image(systemName: "die.face.5")
                        }
                        Spacer()
                    }
                    DisclosureGroup("Backend paths", isExpanded: $showPaths) {
                        VStack(alignment: .leading, spacing: 6) {
                            labeledField("Python", $python)
                            labeledField("Backend dir (app_backend.py)", $backendDir)
                            labeledField("Runtime (bundles + ckpts + ltx_video)", $runtime)
                            labeledField("coreai repo", $coreai)
                            Text("Restart the app after changing paths.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }.padding(.top, 4)
                    }
                }.padding(6)
            }

            switch gen.phase {
            case .loading:
                ProgressView { Text(gen.status).font(.caption) }
                Text("First load reads the bundles (~13.5 GB) — a few seconds.")
                    .font(.caption2).foregroundStyle(.secondary)
            case .generating:
                ProgressView(value: gen.fraction) { Text(gen.status).font(.caption) }
            default:
                Button { gen.generate(prompt: prompt, seed: seed) } label: {
                    Label("Generate video", systemImage: "film")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(gen.phase != .ready || prompt.isEmpty)
            }

            if let err = gen.error {
                ScrollView { Text(err).font(.system(.caption, design: .monospaced)).foregroundStyle(.red) }
                    .frame(maxHeight: 120)
            }

            Spacer()

            if let url = gen.resultURL {
                Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                    Label("Reveal .mp4 in Finder", systemImage: "doc.viewfinder")
                }
            }
        }
    }

    private func labeledField(_ label: String, _ text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField(label, text: text).textFieldStyle(.roundedBorder)
        }
    }

    private var viewer: some View {
        ZStack {
            Color.black
            if let player {
                MacVideoPlayer(player: player)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "film.stack").font(.system(size: 44)).foregroundStyle(.secondary)
                    Text("Your video appears here").foregroundStyle(.secondary)
                }
            }
        }
    }

    private func loadVideo(_ url: URL?) {
        guard let url else { return }
        let p = AVPlayer(url: url)
        p.actionAtItemEnd = .none
        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime,
                                               object: p.currentItem, queue: .main) { _ in
            p.seek(to: .zero); p.play()
        }
        player = p
        p.play()
    }
}
