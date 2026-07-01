import SwiftUI
import SceneKit
import UniformTypeIdentifiers

enum Config {
    // Directory that holds app_backend.py + coreai_out/*.aimodel + ckpts/ + triposplat.py/model.py.
    // Populate it once with setup_runtime.sh; override in Settings if you put it elsewhere.
    static let defaultBackendDir = "~/TripoSplatRuntime"
    static let defaultPython = "~/Code/coreai/coreai-models/.venv/bin/python"
}

struct ContentView: View {
    @StateObject private var gen = Generator()
    @State private var inputURL: URL?
    @State private var inputImage: NSImage?
    @State private var steps = 20
    @State private var scene: SCNScene?
    @State private var showSettings = false
    @AppStorage("python") private var python = Config.defaultPython
    @AppStorage("backendDir") private var backendDir = Config.defaultBackendDir

    var body: some View {
        HSplitView {
            controls
                .frame(minWidth: 320, idealWidth: 340, maxWidth: 420)
                .padding()
            viewer
                .frame(minWidth: 560, minHeight: 560)
        }
        .onChange(of: gen.resultPLY) { _, url in loadScene(url) }
    }

    // MARK: panels

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("TripoSplat · Core AI").font(.title2).bold()
            Text("Image → 3D Gaussian splats, all heavy nets on Apple Core AI.")
                .font(.caption).foregroundStyle(.secondary)

            GroupBox("Input") {
                VStack(spacing: 8) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                        if let img = inputImage {
                            Image(nsImage: img).resizable().scaledToFit().padding(4)
                        } else {
                            Text("No image").foregroundStyle(.secondary)
                        }
                    }
                    .frame(height: 180)
                    Button { pickImage() } label: {
                        Label("Choose Image…", systemImage: "photo")
                    }.frame(maxWidth: .infinity)
                }.padding(6)
            }

            GroupBox("Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    Stepper("Steps: \(steps)", value: $steps, in: 1...50)
                    DisclosureGroup("Backend paths", isExpanded: $showSettings) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Python").font(.caption).foregroundStyle(.secondary)
                            TextField("python", text: $python).textFieldStyle(.roundedBorder)
                            Text("Backend dir").font(.caption).foregroundStyle(.secondary)
                            TextField("backendDir", text: $backendDir).textFieldStyle(.roundedBorder)
                        }.padding(.top, 4)
                    }
                }.padding(6)
            }

            if gen.isRunning {
                ProgressView(value: gen.fraction) { Text(gen.status).font(.caption) }
                Button("Cancel", role: .destructive) { gen.cancel() }
            } else {
                Button { generate() } label: {
                    Label("Generate 3D", systemImage: "cube.transparent")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(inputURL == nil)
            }

            if let err = gen.error {
                ScrollView { Text(err).font(.system(.caption, design: .monospaced)).foregroundStyle(.red) }
                    .frame(maxHeight: 120)
            }

            Spacer()

            if let splat = gen.resultSplat {
                Button { reveal(splat) } label: {
                    Label("Reveal .splat in Finder", systemImage: "doc.viewfinder")
                }
            }
        }
    }

    private var viewer: some View {
        ZStack {
            Color.black
            if let scene {
                SceneView(scene: scene, options: [.allowsCameraControl, .autoenablesDefaultLighting])
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "cube").font(.system(size: 44)).foregroundStyle(.secondary)
                    Text("3D preview appears here").foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: actions

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            inputURL = url
            inputImage = NSImage(contentsOf: url)
            scene = nil
            gen.error = nil
        }
    }

    private func generate() {
        guard let url = inputURL else { return }
        scene = nil
        gen.run(input: url, steps: steps, python: python, backendDir: backendDir)
    }

    private func loadScene(_ url: URL?) {
        guard let url else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            let s = try? PLY.load(url)
            let built = s.map { PLY.scene(from: $0) }
            DispatchQueue.main.async { self.scene = built }
        }
    }

    private func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}
