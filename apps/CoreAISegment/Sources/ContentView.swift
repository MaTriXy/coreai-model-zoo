import SwiftUI
import UniformTypeIdentifiers
import ImageIO
#if canImport(PhotosUI)
import PhotosUI
#endif
#if canImport(AppKit)
import AppKit
#endif

struct ContentView: View {
    @StateObject private var engine = SegmentationEngine()

    @State private var selectedModel = SegmentationEngine.catalog.first
    @State private var prompt = "apple"
    @State private var maxSegments = 5
    @State private var threshold: Float = 0.5
    @State private var showingFolderImporter = false
    @State private var showingImageImporter = false
    #if canImport(PhotosUI)
    @State private var photoItem: PhotosPickerItem?
    #endif

    var body: some View {
        #if os(macOS)
        macBody
        #else
        iosBody
        #endif
    }

    // MARK: - iOS: image fills the screen; prompt + Segment pinned at the bottom (large, demo-friendly)

    #if os(iOS)
    private var iosBody: some View {
        canvas
            .ignoresSafeArea(edges: .bottom)
            .safeAreaInset(edge: .top, spacing: 0) { topBar }
            .safeAreaInset(edge: .bottom, spacing: 0) { bottomBar }
            .photosPicker(isPresented: $showingPhotoPicker, selection: $photoItem, matching: .images)
            .onChange(of: photoItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self) { loadImage(from: data) }
                }
            }
    }

    @State private var showingPhotoPicker = false

    private var topBar: some View {
        HStack(spacing: 12) {
            if let m = selectedModel, engine.canSegment == false {
                Button { engine.loadFromHub(m) } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle.fill")
                        .font(.headline)
                }
                .buttonStyle(.borderedProminent)
                .disabled(engine.status.isBusy)
            }
            if engine.status.isBusy {
                ProgressView().controlSize(.small)
            }
            Text(engine.status.label)
                .font(.subheadline).fontWeight(.medium)
                .foregroundStyle(statusColor).lineLimit(1)
            Spacer()
            if engine.isDownloadingOrLoading {
                Button(role: .destructive) { engine.cancel() } label: { Image(systemName: "xmark.circle.fill") }
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .bottom) {
            if case .downloading = engine.status {
                DownloadBar(downloader: engine.downloader).padding(.horizontal, 16).padding(.bottom, 4)
            }
        }
    }

    private var bottomBar: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button { showingPhotoPicker = true } label: {
                    Image(systemName: "photo.on.rectangle.angled").font(.title2)
                }
                .buttonStyle(.bordered)

                TextField("what to segment — e.g. apple", text: $prompt)
                    .font(.title3)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .submitLabel(.go)
                    .onSubmit(runSegment)
            }
            Button(action: runSegment) {
                Label("Segment", systemImage: "scribble.variable")
                    .font(.title3).fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .disabled(!engine.canSegment || engine.sourceImage == nil
                      || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.horizontal, 16).padding(.top, 10).padding(.bottom, 8)
        .background(.bar)
    }
    #endif

    // MARK: - macOS: split view

    #if os(macOS)
    private var macBody: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    group("Model") { modelControls }
                    group("Image") { imageControls }
                    group("Prompt") { promptControls }
                    group("Settings") { settingsControls }
                    segmentButton
                }
                .padding(18)
            }
            .frame(minWidth: 320, idealWidth: 350, maxWidth: 440)
            canvas.frame(minWidth: 480)
        }
        .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { engine.loadLocal(url) }
        }
        .fileImporter(isPresented: $showingImageImporter, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result { loadImage(from: url) }
        }
    }

    @ViewBuilder private var modelControls: some View {
        HStack {
            if let model = selectedModel {
                Button { engine.loadFromHub(model) } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle")
                }.disabled(engine.status.isBusy)
            }
            Button {
                if engine.isDownloadingOrLoading { engine.cancel() }
                showingFolderImporter = true
            } label: { Label("Local…", systemImage: "folder") }
        }
        if engine.isDownloadingOrLoading {
            Button(role: .destructive) { engine.cancel() } label: { Label("Cancel", systemImage: "xmark.circle") }
        }
        statusLine
    }
    @ViewBuilder private var imageControls: some View {
        Button { showingImageImporter = true } label: { Label("Choose Image…", systemImage: "photo") }
    }
    @ViewBuilder private var promptControls: some View {
        TextField("What to segment (e.g. apple)", text: $prompt).textFieldStyle(.roundedBorder)
    }
    private var segmentButton: some View {
        Button(action: runSegment) {
            Label("Segment", systemImage: "scribble.variable").frame(maxWidth: .infinity)
        }
        .controlSize(.large).buttonStyle(.borderedProminent)
        .disabled(!engine.canSegment || engine.sourceImage == nil
                  || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
    }
    @ViewBuilder
    private func group<C: View>(_ title: String, @ViewBuilder _ c: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            c()
        }
    }
    #endif

    @ViewBuilder private var settingsControls: some View {
        Stepper("Max segments: \(maxSegments)", value: $maxSegments, in: 1...20)
        VStack(alignment: .leading, spacing: 4) {
            HStack { Text("Mask threshold"); Spacer()
                Text(String(format: "%.2f", threshold)).monospacedDigit().foregroundStyle(.secondary) }
            Slider(value: $threshold, in: 0.05...0.95)
        }
    }

    private var statusLine: some View {
        HStack(spacing: 8) {
            if engine.status.isBusy { ProgressView().controlSize(.small) }
            Text(engine.status.label).font(.caption).foregroundStyle(statusColor).lineLimit(2)
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusColor: Color {
        if case .error = engine.status { return .red }
        if case .ready = engine.status { return .green }
        return .secondary
    }

    private func runSegment() {
        engine.segment(prompt: prompt, maxSegments: maxSegments, threshold: threshold)
    }

    // MARK: - Canvas (shared)

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let cg = engine.resultImage ?? engine.sourceImage {
                Image(decorative: cg, scale: 1)
                    .resizable().interpolation(.high)
                    .aspectRatio(contentMode: .fit)
            } else {
                placeholder
            }
            if case .segmenting = engine.status {
                ProgressView().controlSize(.large).tint(.white)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay(alignment: .topTrailing) {
            if let secs = engine.segmentSeconds, engine.resultImage != nil {
                Text("\(engine.segmentCount) segment\(engine.segmentCount == 1 ? "" : "s") · \(String(format: "%.2fs", secs))")
                    .font(.callout).fontWeight(.medium).foregroundStyle(.white)
                    .padding(.horizontal, 12).padding(.vertical, 7)
                    .background(.black.opacity(0.55), in: Capsule()).padding(12)
            }
        }
    }

    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "scribble.variable").font(.system(size: 54)).foregroundStyle(.white.opacity(0.5))
            Text(placeholderText).font(.title3).foregroundStyle(.white.opacity(0.75))
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle: return "Tap Download & Load, then pick a photo."
        case .downloading: return "Downloading SAM 3 — ~1.7 GB, cached after first run."
        case .loading: return "Loading SAM 3…"
        case .ready: return engine.sourceImage == nil
            ? "Pick a photo, type what to segment, tap Segment."
            : "Type what to segment and tap Segment."
        case .segmenting: return ""
        case .error(let m): return m
        }
    }

    // MARK: - Image loading

    private func loadImage(from url: URL) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }
        engine.setImage(cg)
    }
    private func loadImage(from data: Data) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }
        engine.setImage(cg)
    }
}

private struct DownloadBar: View {
    @ObservedObject var downloader: ModelDownloader
    var body: some View {
        VStack(spacing: 4) {
            ProgressView(value: downloader.fraction)
            Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}
