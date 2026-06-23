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
    @State private var prompt = "cat"
    @State private var maxSegments = 5
    @State private var threshold: Float = 0.5
    @State private var showingFolderImporter = false
    @State private var showingImageImporter = false
    #if canImport(PhotosUI)
    @State private var photoItem: PhotosPickerItem?
    #endif

    var body: some View {
        #if os(macOS)
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
        #else
        NavigationStack {
            GeometryReader { geo in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        canvas
                            .frame(maxWidth: .infinity)
                            .frame(height: 280)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        card("Model") { modelControls }
                        card("Image") { imageControls }
                        card("Prompt") { promptControls }
                        card("Settings") { settingsControls }
                    }
                    .padding(16)
                    .frame(width: geo.size.width, alignment: .leading)
                }
                .scrollDismissesKeyboard(.interactively)
                .safeAreaInset(edge: .bottom) {
                    segmentButton
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(.bar)
                }
            }
            .navigationTitle("CoreAI Segment")
            .navigationBarTitleDisplayMode(.inline)
        }
        #endif
    }

    // MARK: - Controls

    @ViewBuilder private var modelControls: some View {
        HStack {
            if let model = selectedModel {
                Button { engine.loadFromHub(model) } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle")
                }
                .disabled(engine.status.isBusy)
            }
            Button {
                if engine.isDownloadingOrLoading { engine.cancel() }
                showingFolderImporter = true
            } label: { Label("Local…", systemImage: "folder") }
        }
        if engine.isDownloadingOrLoading {
            Button(role: .destructive) { engine.cancel() } label: {
                Label("Cancel", systemImage: "xmark.circle")
            }
        }
        statusLine
    }

    @ViewBuilder private var imageControls: some View {
        #if canImport(PhotosUI) && os(iOS)
        PhotosPicker(selection: $photoItem, matching: .images) {
            Label("Choose Photo", systemImage: "photo")
        }
        .onChange(of: photoItem) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self) {
                    loadImage(from: data)
                }
            }
        }
        #else
        Button { showingImageImporter = true } label: {
            Label("Choose Image…", systemImage: "photo")
        }
        #endif
    }

    @ViewBuilder private var promptControls: some View {
        TextField("What to segment (e.g. cat)", text: $prompt)
            .textFieldStyle(.roundedBorder)
            #if os(iOS)
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)
            #endif
    }

    @ViewBuilder private var settingsControls: some View {
        Stepper("Max segments: \(maxSegments)", value: $maxSegments, in: 1...20)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Mask threshold")
                Spacer()
                Text(String(format: "%.2f", threshold)).monospacedDigit().foregroundStyle(.secondary)
            }
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

    private var segmentButton: some View {
        Button {
            engine.segment(prompt: prompt, maxSegments: maxSegments, threshold: threshold)
        } label: {
            Label("Segment", systemImage: "scribble.variable").frame(maxWidth: .infinity)
        }
        .controlSize(.large)
        .buttonStyle(.borderedProminent)
        .disabled(!engine.canSegment || engine.sourceImage == nil
                  || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    // MARK: - Canvas

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let cg = engine.resultImage ?? engine.sourceImage {
                Image(decorative: cg, scale: 1)
                    .resizable().interpolation(.high)
                    .aspectRatio(contentMode: .fit).padding(12)
            } else {
                placeholder
            }
            if case .downloading = engine.status {
                VStack { Spacer()
                    DownloadBar(downloader: engine.downloader)
                        .padding(12).background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10)).padding(20)
                }
            } else if case .segmenting = engine.status {
                ProgressView().controlSize(.large)
            }
        }
        .overlay(alignment: .topTrailing) {
            if let secs = engine.segmentSeconds, engine.resultImage != nil {
                Text("\(engine.segmentCount) segment\(engine.segmentCount == 1 ? "" : "s") · \(String(format: "%.2fs", secs))")
                    .font(.caption).foregroundStyle(.white.opacity(0.8))
                    .padding(8).background(.black.opacity(0.4), in: Capsule()).padding(10)
            }
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "scribble.variable").font(.system(size: 46)).foregroundStyle(.tertiary)
            Text(placeholderText).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle: return "Tap Download & Load to fetch SAM 3, then choose an image."
        case .downloading: return "Downloading the converted bundle from Hugging Face — ~1.7 GB, cached after the first run."
        case .loading: return "Loading SAM 3 into the Core AI runtime…"
        case .ready: return engine.sourceImage == nil
            ? "Choose an image, type what to segment, and tap Segment."
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

    // MARK: - Layout helpers

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
    }

    #if os(iOS)
    @ViewBuilder
    private func card<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased()).font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }
    #endif
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
