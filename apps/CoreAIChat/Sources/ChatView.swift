import PhotosUI
import SwiftUI

struct Turn: Identifiable {
    let id = UUID()
    let role: String      // "user" | "model"
    var text: String
    var stats: String = ""
    var image: UIImage? = nil
}

struct ChatView: View {
    @ObservedObject var engine: Gemma4ChatEngine
    @StateObject private var downloader = ModelDownloader()
    @State private var input = ""
    @State private var turns: [Turn] = []
    @State private var repoURL = ProcessInfo.processInfo.environment["GEMMA_REPO"]
        ?? Gemma4ChatEngine.defaultRepo(for: .gpu)
    // Remembered gemma compute unit, so Gemma -> Qwen -> Gemma comes back to
    // the unit the user had (the pipelined models have no unit choice).
    @State private var gemmaUnit: GemmaMode = .gpu
    @FocusState private var inputFocused: Bool
    @State private var photoItem: PhotosPickerItem?
    @State private var attachedThumb: UIImage?
    @State private var showDeleteConfirm = false

    // Two-level selection over the flat engine mode: model (menu) x unit
    // (gemma-only segment).
    private var modelSelection: Binding<ChatModel> {
        Binding(
            get: { engine.mode.chatModel },
            set: { m in
                switch m {
                case .gemma: engine.mode = gemmaUnit
                case .qwen: engine.mode = .qwen
                case .qwen2b: engine.mode = .qwen2b
                case .lfm2: engine.mode = .lfm2
                case .granite: engine.mode = .granite
                case .minicpm5: engine.mode = .minicpm5
                case .qwen3vl: engine.mode = .qwen3vl
                case .qwen3vl4b: engine.mode = .qwen3vl4b
                case .gemma4vl: engine.mode = .gemma4vl
                }
            })
    }

    private var unitSelection: Binding<GemmaMode> {
        Binding(
            get: {
                switch engine.mode {
                case .ane: .ane
                case .gemmaTbl: .gemmaTbl
                default: .gpu
                }
            },
            set: { u in
                gemmaUnit = u
                engine.mode = u
            })
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(turns) { turn in bubble(turn) }
                    // Live model output while generating.
                    if engine.busy {
                        bubble(Turn(role: "model", text: engine.output.isEmpty ? "…" : engine.output,
                                    stats: engine.stats))
                            .id("live")
                    }
                }
                .padding()
            }
            .onChange(of: engine.output) { _, _ in proxy.scrollTo("live", anchor: .bottom) }
            .scrollDismissesKeyboard(.interactively)
            // Liquid Glass: floating bars; chat content scrolls underneath the glass.
            .safeAreaInset(edge: .top) { topBar }
            .safeAreaInset(edge: .bottom) { inputBar }
        }
        // Mode switch (gemma GPU monolith / ANE chunks / ⚡tbl, or a pipelined model): point the
        // download field at that mode's HF repo (unless GEMMA_REPO pins it) and reload.
        .onChange(of: engine.mode) { _, m in
            if !m.isVL { photoItem = nil; attachedThumb = nil }
            if ProcessInfo.processInfo.environment["GEMMA_REPO"] == nil {
                repoURL = Gemma4ChatEngine.defaultRepo(for: m)
            }
            Task { await engine.load() }
        }
        // Raise the keyboard on launch (typing is allowed while the model loads; Send stays gated).
        .onAppear {
            inputFocused = true
            // headless GEMMA_ENGINE=ane/gemmatbl start: keep the segment in sync
            if engine.mode.chatModel == .gemma { gemmaUnit = engine.mode }
            if ProcessInfo.processInfo.environment["GEMMA_REPO"] == nil {
                repoURL = Gemma4ChatEngine.defaultRepo(for: engine.mode)
            }
        }
        .confirmationDialog("Delete \(engine.mode.downloadLabel) files?",
                            isPresented: $showDeleteConfirm, titleVisibility: .visible) {
            Button("Delete", role: .destructive) { Task { await engine.deleteCurrentModel() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Frees space on this iPhone. You can download it again anytime.")
        }
    }

    private var topBar: some View {
        GlassEffectContainer(spacing: 12) {
            VStack(spacing: 12) {
                header
                // Model delivery: the load failed (or never ran) and files for this mode are missing.
                if !engine.ready && !engine.loading {
                    let missing = engine.missingDownloads()
                    if !missing.isEmpty { downloadPanel(missing) }
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("CoreAIChat").font(.headline)
            // Model first (a menu scales as the zoo grows) ...
            Picker("Model", selection: modelSelection) {
                ForEach(ChatModel.allCases) { m in Text(m.rawValue).tag(m) }
            }
            .pickerStyle(.menu)
            .disabled(engine.busy || engine.loading)
            Spacer()
            // ... then the engine, only where there is a choice (gemma:
            // GPU kernel monolith / ANE chunks / ⚡pipelined static-table).
            if engine.mode.chatModel == .gemma {
                Picker("Compute", selection: unitSelection) {
                    Text("GPU").tag(GemmaMode.gpu)
                    Text("ANE").tag(GemmaMode.ane)
                    Text("⚡").tag(GemmaMode.gemmaTbl)
                }
                .pickerStyle(.segmented)
                .frame(width: 148)
                .disabled(engine.busy || engine.loading)
            }
            Text(engine.status)
                .font(.caption).foregroundStyle(engine.ready ? .green : .secondary)
                .lineLimit(1)
            // Free this model's files from the device (re-downloadable). Only when it's installed.
            if engine.installedForCurrentMode {
                Button(role: .destructive) { showDeleteConfirm = true } label: {
                    Image(systemName: "trash")
                }
                .disabled(engine.busy || engine.loading)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .glassEffect()
        .padding(.horizontal, 8)
    }

    private func bubble(_ turn: Turn) -> some View {
        HStack {
            if turn.role == "user" { Spacer(minLength: 40) }
            VStack(alignment: turn.role == "user" ? .trailing : .leading, spacing: 4) {
                if let image = turn.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 340)
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                }
                if turn.image == nil || !turn.text.isEmpty {
                    Text(turn.text.isEmpty ? " " : turn.text)
                        .textSelection(.enabled)
                        .padding(.horizontal, 14).padding(.vertical, 9)
                        .foregroundStyle(turn.role == "user" ? .white : .primary)
                        .background(turn.role == "user" ? Color.accentColor : Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                }
                if !turn.stats.isEmpty {
                    Text(turn.stats).font(.caption2).foregroundStyle(.secondary)
                }
            }
            if turn.role == "model" { Spacer(minLength: 40) }
        }
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if engine.mode.isVL {
                PhotosPicker(selection: $photoItem, matching: .images) {
                    if let attachedThumb {
                        Image(uiImage: attachedThumb)
                            .resizable().scaledToFill()
                            .frame(width: 34, height: 34)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(alignment: .topTrailing) {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 12)).foregroundStyle(.green)
                            }
                    } else {
                        Image(systemName: "photo.badge.plus").font(.system(size: 24))
                    }
                }
                .disabled(!engine.ready || engine.busy)
                .padding(.bottom, 6)
                .onChange(of: photoItem) { item in
                    guard let item else { return }
                    Task {
                        if let data = try? await item.loadTransferable(type: Data.self),
                           let ui = UIImage(data: data), let cg = ui.cgImage {
                            attachedThumb = ui
                            turns.append(Turn(role: "user", text: "", image: ui))
                            await engine.attachVLImage(cg)
                        }
                    }
                }
            }
            TextField("Message", text: $input, axis: .vertical)
                .lineLimit(1...4)
                .focused($inputFocused)
                .padding(.horizontal, 14).padding(.vertical, 9)
                .glassEffect(in: .rect(cornerRadius: 20))
            Button {
                Task { await send() }
            } label: {
                Image(systemName: "arrow.up.circle.fill").font(.system(size: 31))
            }
            .disabled(!engine.ready || engine.busy || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .padding(.bottom, 4)
        }
        .padding(.horizontal).padding(.vertical, 8)
    }

    private func send() async {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        input = ""
        // Tuck the keyboard away while generating; tapping the field brings it back.
        inputFocused = false
        turns.append(Turn(role: "user", text: prompt))
        await engine.generate(prompt)
        turns.append(Turn(role: "model", text: engine.output, stats: engine.stats))
    }

    // MARK: model delivery (published artifact set from the Hugging Face repo)

    private func downloadPanel(_ items: [ModelDownloader.Item]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(engine.mode.downloadLabel) model files not on device (\(items.count))")
                .font(.subheadline).fontWeight(.semibold)
            TextField("Hugging Face repo URL", text: $repoURL)
                .textFieldStyle(.roundedBorder).font(.caption)
                .autocorrectionDisabled().textInputAutocapitalization(.never)
                .disabled(downloader.busy)
            if downloader.busy {
                ProgressView(value: downloader.fraction)
                Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Button {
                    Task {
                        await downloader.fetch(
                            repo: repoURL, items: items,
                            into: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                                .appendingPathComponent("models"))
                        if downloader.phase == .done { await engine.load() }
                    }
                } label: {
                    let size = switch engine.mode {
                    case .gpu: "~4.1"
                    case .ane: "~2.1–4.7"
                    case .gemmaTbl: "~2.0–4.8"
                    case .qwen: "~1.3"
                    case .qwen2b: "~2.4"
                    case .lfm2: "~1.5"
                    case .granite: "~1.2"
                    case .minicpm5: "~2.0"
                    case .qwen3vl: "~3.1"
                    case .qwen3vl4b: "~5.5"
                    case .gemma4vl: "~4.7"
                    }
                    Label("Download \(engine.mode.downloadLabel) set (\(size) GB)",
                          systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            }
            if case .failed(let msg) = downloader.phase {
                Text(msg).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(10)
        .glassEffect(in: .rect(cornerRadius: 16))
        .padding(.horizontal)
    }
}
