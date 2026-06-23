// CoreAIOCRApp.swift — minimal macOS app: pick a document image -> on-device OCR markdown.
// Self-contained shell around OCRPipeline. Build with xcodegen (project.yml) + Xcode-beta,
// run on device (the harness can't build macOS apps). Place the assets in `assetsDir`
// (out/_swift_assets/* + the two .aimodel bundles + a tokenizer/ dir).

import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct CoreAIOCRApp: App {
    var body: some Scene { WindowGroup { ContentView() }.defaultSize(width: 980, height: 720) }
}

@MainActor
final class OCRViewModel: ObservableObject {
    @Published var image: NSImage?
    @Published var markdown = ""
    @Published var status = "Load model, then drop a document image."
    @Published var busy = false
    private var pipeline: OCRPipeline?

    /// Folder holding: _vision_export.aimodel, _dec_unified.aimodel, embed_tokens.f16,
    /// image_newline.f16, view_seperator.f16, prompt_input_ids.i32, tokenizer/.
    var assetsDir: URL = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("CoreAIOCR", isDirectory: true)

    func loadModel() async {
        busy = true; status = "Loading model…"
        do {
            let a = ModelAssets(
                visionAIModel: assetsDir.appendingPathComponent("_vision_export.aimodel"),
                decoderAIModel: assetsDir.appendingPathComponent("_dec_unified.aimodel"),
                embedTokens: assetsDir.appendingPathComponent("embed_tokens.f16"),
                imageNewline: assetsDir.appendingPathComponent("image_newline.f16"),
                viewSeperator: assetsDir.appendingPathComponent("view_seperator.f16"),
                promptInputIDs: assetsDir.appendingPathComponent("prompt_input_ids.i32"),
                tokenizerDir: assetsDir.appendingPathComponent("tokenizer", isDirectory: true))
            pipeline = try await OCRPipeline(assets: a)
            status = "Model ready. Drop a document image."
        } catch { status = "Load failed: \(error)" }
        busy = false
    }

    /// Verification convenience: if `sample.png` is staged next to the assets, load the
    /// model and OCR it automatically on launch. Normal use (no sample.png) is manual.
    func autostartIfSample() async {
        let sample = assetsDir.appendingPathComponent("sample.png")
        guard FileManager.default.fileExists(atPath: sample.path) else { return }
        await loadModel()
        if let ns = NSImage(contentsOf: sample) { await run(ns) }
    }

    /// Reliable image input: a file picker (drag-and-drop from Finder is finicky).
    func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await recognize(url: url) }
    }

    /// Load an image from a URL (Sendable across the Task boundary) and OCR it.
    func recognize(url: URL) async {
        guard let ns = NSImage(contentsOf: url) else { status = "Can't read image at \(url.lastPathComponent)"; return }
        await run(ns)
    }

    func run(_ ns: NSImage) async {
        if pipeline == nil { await loadModel() }     // lazy-load on first image
        guard let pipeline else { status = "Load the model first."; return }
        guard let cg = ns.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return }
        image = ns; busy = true; status = "Recognizing…"; markdown = ""
        do {
            let t = Date()
            markdown = try await pipeline.recognize(cg)
            status = String(format: "Done in %.1fs", Date().timeIntervalSince(t))
        } catch { status = "OCR failed: \(error)" }
        busy = false
    }
}

struct ContentView: View {
    @StateObject private var vm = OCRViewModel()

    var body: some View {
        HSplitView {
            VStack {
                if let img = vm.image {
                    Image(nsImage: img).resizable().scaledToFit()
                } else {
                    RoundedRectangle(cornerRadius: 12).strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
                        .overlay(Text("Drop a document image").foregroundStyle(.secondary))
                }
            }
            .frame(minWidth: 360).padding()
            .onTapGesture { vm.openImage() }   // click the drop zone to pick a file
            .onDrop(of: [.fileURL, .image], isTargeted: nil) { providers in
                guard let p = providers.first else { return false }
                if p.canLoadObject(ofClass: URL.self) {
                    _ = p.loadObject(ofClass: URL.self) { url, _ in
                        if let url { Task { await vm.recognize(url: url) } }
                    }
                    return true
                }
                _ = p.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                    guard let data, let url = Self.writeTemp(data) else { return }
                    Task { await vm.recognize(url: url) }
                }
                return true
            }
            ScrollView { Text(vm.markdown).font(.system(.body, design: .monospaced))
                .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding() }
                .frame(minWidth: 380)
        }
        .toolbar {
            ToolbarItem { Button("Open Image…") { vm.openImage() }.disabled(vm.busy) }
            ToolbarItem { Button("Load Model") { Task { await vm.loadModel() } }.disabled(vm.busy) }
            ToolbarItem { if vm.busy { ProgressView().controlSize(.small) } }
        }
        .safeAreaInset(edge: .bottom) { Text(vm.status).font(.caption).foregroundStyle(.secondary).padding(6) }
        .task { await vm.autostartIfSample() }
    }

    static func writeTemp(_ data: Data) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".img")
        return (try? data.write(to: url)) == nil ? nil : url
    }
}
