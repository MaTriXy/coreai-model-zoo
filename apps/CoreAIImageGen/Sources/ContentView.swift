import CoreGraphics
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

/// Generation mode. Text→Image (pure prompt), Image→Image (SDEdit restyle with strength), and
/// Edit (FLUX.2 in-context edit: attend to a reference image, prompt is the edit instruction).
private enum GenMode: String, CaseIterable, Identifiable {
    case textToImage = "Text → Image"
    case imageToImage = "Image → Image"
    case edit = "Edit"
    var id: String { rawValue }
    /// True when this mode needs a source/reference image.
    var usesImage: Bool { self != .textToImage }
    var actionLabel: String {
        switch self {
        case .textToImage: return "Generate"
        case .imageToImage: return "Transform"
        case .edit: return "Apply Edit"
        }
    }
    var actionIcon: String {
        switch self {
        case .textToImage: return "sparkles"
        case .imageToImage: return "wand.and.stars"
        case .edit: return "wand.and.rays"
        }
    }
}

struct ContentView: View {
    @StateObject private var engine = DiffusionEngine()

    // Optional: the iOS build ships an empty hosted catalog (load via "Local…").
    @State private var selectedModel = DiffusionEngine.defaultSelection
    @State private var prompt = "a watercolor painting of a red fox reading a book by candlelight, cozy, detailed"
    @State private var negativePrompt = ""
    @State private var steps = DiffusionEngine.catalog.first?.defaultSteps ?? 4
    @State private var guidance = DiffusionEngine.catalog.first?.defaultGuidance ?? 1.0
    @State private var seedText = "42"
    @State private var showingFolderImporter = false

    // Image-to-image. Default 0.85 matches the runtime's own default: a guidance-distilled
    // 4-step model needs a high strength to actually change the image — below ~0.6 the
    // denoiser mostly reconstructs the source, which reads as "nothing happened".
    @State private var mode: GenMode = .textToImage
    @State private var inputImage: CGImage?
    @State private var strength: Float = 0.85
    @State private var showingImageImporter = false
    // Edit mode: an optional second reference image (combine / place subject across images).
    @State private var inputImage2: CGImage?
    @State private var showingImageImporter2 = false

    /// Modes the loaded model supports: Image→Image needs a VAE encoder, Edit needs the
    /// edit-sequence transformer. Text→Image is always available.
    private var availableModes: [GenMode] {
        var m: [GenMode] = [.textToImage]
        if engine.supportsImg2Img { m.append(.imageToImage) }
        if engine.supportsEdit || engine.canDownloadEditAssets { m.append(.edit) }
        return m
    }

    var body: some View {
        #if os(macOS)
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    group("Model") { modelControls }
                    if availableModes.count > 1 { group("Mode") { modeControls } }
                    group(mode == .edit ? "Instruction" : "Prompt") { promptControls(maxWidth: nil) }
                    group("Settings") { settingsControls }
                    generateButton
                }
                .padding(18)
            }
            .frame(minWidth: 320, idealWidth: 350, maxWidth: 440)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
            .fileImporter(isPresented: $showingImageImporter, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url, into: 1) }
            }
            .fileImporter(isPresented: $showingImageImporter2, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url, into: 2) }
            }
            .onChange(of: engine.supportsImg2Img) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
            .onChange(of: engine.supportsEdit) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
            canvas.frame(minWidth: 460)
        }
        #else
        NavigationStack {
            // GeometryReader gives a CONCRETE content width. A `TextField(axis: .vertical)`
            // reports its full single-line intrinsic width upward, which inside a Form/List
            // blows the whole column wider than the screen (content overflows both edges).
            // Capping the text fields at this measured width — not `.infinity` — keeps the
            // reported width bounded, so everything stays inside the device.
            GeometryReader { geo in
                let contentWidth = geo.size.width - 32
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        canvas
                            .frame(maxWidth: .infinity)
                            .frame(height: 230)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        card("Model") { modelControls }
                        if availableModes.count > 1 { card("Mode") { modeControls } }
                        card(mode == .edit ? "Instruction" : "Prompt") { promptControls(maxWidth: contentWidth) }
                        card("Settings") { settingsControls }
                    }
                    .padding(16)
                    .frame(width: geo.size.width, alignment: .leading)
                }
                .scrollDismissesKeyboard(.interactively)
                .safeAreaInset(edge: .bottom) {
                    generateButton
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.bar)
                }
            }
            .navigationTitle("CoreAI Image Gen")
            .navigationBarTitleDisplayMode(.inline)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
            .fileImporter(isPresented: $showingImageImporter, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url, into: 1) }
            }
            .fileImporter(isPresented: $showingImageImporter2, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url, into: 2) }
            }
            .onChange(of: engine.supportsImg2Img) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
            .onChange(of: engine.supportsEdit) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
        }
        #endif
    }

    // MARK: - Control content (shared; wrapped in `group`/`card` per platform)

    @ViewBuilder private var modelControls: some View {
        if !DiffusionEngine.catalog.isEmpty {
            Picker("Model", selection: $selectedModel) {
                ForEach(DiffusionEngine.catalog) { Text($0.title).tag(Optional($0)) }
            }
            #if os(macOS)
            .labelsHidden()
            #else
            .pickerStyle(.menu)
            #endif
            .disabled(engine.status.isBusy)
        }

        HStack {
            if let model = selectedModel {
                Button {
                    steps = model.defaultSteps
                    guidance = model.defaultGuidance
                    engine.loadFromHub(model)
                } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle")
                }
                .disabled(engine.status.isBusy)
            }
            // Local… stays enabled even mid-download — tapping it cancels the transfer first.
            Button {
                if engine.isDownloadingOrLoading { engine.cancel() }
                showingFolderImporter = true
            } label: {
                Label("Local…", systemImage: "folder")
            }
        }

        if engine.isDownloadingOrLoading {
            Button(role: .destructive) { engine.cancel() } label: {
                Label("Cancel download", systemImage: "xmark.circle")
            }
        }

        statusLine
    }

    // Mode switch + (in image-to-image) the source picker and strength. Shown only when the
    // loaded model has a VAE encoder — FLUX.2 bundles do; a bare txt2img model would not.
    @ViewBuilder private var modeControls: some View {
        Picker("Mode", selection: $mode) {
            ForEach(availableModes) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .disabled(engine.status.isBusy)

        if mode.usesImage {
            if mode == .edit && !engine.supportsEdit {
                editDownloadPrompt
            } else {
                sourceImageControls
            }
        }
    }

    /// Shown in Edit mode when the edit transformers aren't downloaded yet (hosted model).
    @ViewBuilder private var editDownloadPrompt: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { engine.downloadEditAssets() } label: {
                Label("Download edit models (~4 GB)", systemImage: "arrow.down.circle")
            }
            .disabled(engine.status.isBusy)
            Text("In-context editing uses separate transformers, fetched on demand.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var sourceImageControls: some View {
        // Source preview — full width at its true aspect ratio, tap to (re)choose.
        Group {
            if let cg = inputImage {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .frame(maxHeight: 240)
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(white: 0.12))
                    .frame(height: 120)
                    .overlay {
                        VStack(spacing: 6) {
                            Image(systemName: "photo.badge.plus").font(.title2)
                            Text("Choose a source image").font(.caption)
                        }
                        .foregroundStyle(.secondary)
                    }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .contentShape(Rectangle())
        .onTapGesture { if !engine.status.isBusy { showingImageImporter = true } }

        Button { showingImageImporter = true } label: {
            Label(inputImage == nil ? "Choose Image…" : "Change Image…", systemImage: "photo")
                .frame(maxWidth: .infinity)
        }
        .disabled(engine.status.isBusy)

        if mode == .imageToImage {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Strength")
                    Spacer()
                    Text(String(format: "%.2f", strength)).monospacedDigit().foregroundStyle(.secondary)
                }
                Slider(value: $strength, in: 0.3...1.0)
                Text("0.8–0.9 for content edits · 0.5–0.75 for style/texture · lower keeps the source.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        } else if mode == .edit {
            if engine.supports2refEdit {
                HStack(spacing: 10) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 8).fill(Color(white: 0.12))
                        if let cg = inputImage2 {
                            Image(decorative: cg, scale: 1).resizable().aspectRatio(contentMode: .fit)
                        } else {
                            Image(systemName: "photo.badge.plus").foregroundStyle(.secondary)
                        }
                    }
                    .frame(width: 52, height: 52)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    Button { showingImageImporter2 = true } label: {
                        Label(inputImage2 == nil ? "Add 2nd reference…" : "Change 2nd", systemImage: "photo.on.rectangle")
                    }
                    .disabled(engine.status.isBusy)
                    if inputImage2 != nil {
                        Button { inputImage2 = nil } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.borderless).foregroundStyle(.secondary)
                    }
                    Spacer()
                }
            }
            Text("References are kept; the instruction edits/combines them (e.g. \"add a red hat\", or with a 2nd reference \"put the subject into this scene\").")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func promptControls(maxWidth: CGFloat?) -> some View {
        TextField("Prompt", text: $prompt, axis: .vertical)
            .lineLimit(2...6)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
        TextField("Negative prompt (optional)", text: $negativePrompt, axis: .vertical)
            .lineLimit(1...3)
            .foregroundStyle(.secondary)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
    }

    @ViewBuilder private var settingsControls: some View {
        Stepper("Steps: \(steps)", value: $steps, in: 1...50)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Guidance")
                Spacer()
                Text(String(format: "%.1f", guidance)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $guidance, in: 0...10)
        }
        HStack {
            Text("Seed")
            TextField("seed", text: $seedText)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif
            Button { seedText = String(UInt32.random(in: 0 ... .max)) } label: {
                Image(systemName: "die.face.5")
            }
            .buttonStyle(.borderless)
        }
    }

    private var statusLine: some View {
        HStack(spacing: 8) {
            if engine.status.isBusy { ProgressView().controlSize(.small) }
            Text(engine.status.label)
                .font(.caption).foregroundStyle(statusColor).lineLimit(2)
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusColor: Color {
        if case .error = engine.status { return .red }
        if case .ready = engine.status { return .green }
        return .secondary
    }

    private var generateButton: some View {
        Group {
            if case .generating = engine.status {
                Button(role: .destructive) { engine.cancel() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
            } else {
                Button {
                    let seedValue = UInt32(seedText) ?? 42
                    if mode == .edit, let ref = inputImage {
                        var refs = [ref]
                        if engine.supports2refEdit, let ref2 = inputImage2 { refs.append(ref2) }
                        engine.edit(referenceImages: refs, instruction: prompt, steps: steps, seed: seedValue)
                    } else {
                        engine.generate(
                            prompt: prompt, negativePrompt: negativePrompt,
                            steps: steps, guidance: guidance, seed: seedValue,
                            startingImage: mode == .imageToImage ? inputImage : nil,
                            strength: strength)
                    }
                } label: {
                    Label(mode.actionLabel, systemImage: mode.actionIcon).frame(maxWidth: .infinity)
                }
                .disabled(!engine.canGenerate
                    || (mode != .imageToImage && prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                    || (mode.usesImage && inputImage == nil)
                    || (mode == .edit && !engine.supportsEdit))
            }
        }
        .controlSize(.large)
        .buttonStyle(.borderedProminent)
    }

    // MARK: - Canvas (shared)

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let cg = engine.image {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .padding(12)
            } else {
                placeholder
            }

            if case .generating(let s, let t) = engine.status {
                progressOverlay(value: Double(s), total: Double(max(t, 1)))
            } else if case .downloading = engine.status {
                VStack {
                    Spacer()
                    DownloadBar(downloader: engine.downloader)
                        .padding(12)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
                        .padding(20)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 12) {
                if let secs = engine.generateSeconds, engine.image != nil {
                    Text("\(engine.imageSize) · \(String(format: "%.1fs", secs))")
                        .font(.caption).foregroundStyle(.white.opacity(0.7))
                }
                if let url = engine.exportURL {
                    Button {
                        if let saved = engine.saveImageToDownloads() { reveal(saved) }
                    } label: {
                        Image(systemName: engine.savedURL != nil
                              ? "checkmark.circle.fill" : "square.and.arrow.down")
                    }
                    .help("Save the image to Downloads")
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
            }
            .buttonStyle(.borderless)
            .padding(10)
        }
    }

    /// Decode a user-picked image file into a CGImage for the image-to-image source.
    /// The runtime resizes it to the model's native size, so any resolution is fine.
    private func loadInputImage(from url: URL, into slot: Int = 1) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }
        if slot == 2 { inputImage2 = cg } else { inputImage = cg }
    }

    /// Reveal a saved file in Finder (macOS); no-op elsewhere.
    private func reveal(_ url: URL) {
        #if os(macOS)
        NSWorkspace.shared.activateFileViewerSelecting([url])
        #endif
    }

    private func progressOverlay(value: Double, total: Double) -> some View {
        VStack {
            Spacer()
            ProgressView(value: value, total: total) {
                Text(engine.status.label).font(.caption)
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
            .padding(20)
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "photo.artframe")
                .font(.system(size: 46)).foregroundStyle(.tertiary)
            Text(placeholderText)
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle:
            return DiffusionEngine.catalog.isEmpty
                ? "Tap Local… to load a Core AI diffusion bundle (e.g. Stable Diffusion)."
                : "Pick a model and tap Download & Load to begin."
        case .downloading: return "Downloading the converted bundle from Hugging Face — a few GB, cached after the first run."
        case .loading: return "Loading the model into the Core AI runtime…"
        case .ready: return "Enter a prompt and tap Generate."
        case .error(let m): return m
        case .generating: return ""
        }
    }

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
    }

    #if os(iOS)
    @ViewBuilder
    private func card<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 14))
    }
    #endif
}

/// Download progress, observing the shared downloader directly so the bar and byte
/// counter advance as chunks land (the engine only owns the high-level phase).
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
