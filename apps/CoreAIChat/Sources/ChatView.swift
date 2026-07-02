import PhotosUI
import SwiftUI

/// Animated grounding marker: a spring "pop" landing + a repeating radar ping ring,
/// so the located UI element is obvious in a demo video.
struct GroundMarker: View {
    @State private var ping = false
    @State private var landed = false
    var body: some View {
        ZStack {
            // expanding radar ping (repeats forever)
            Circle().stroke(.red, lineWidth: 3).frame(width: 50, height: 50)
                .scaleEffect(ping ? 2.8 : 0.6)
                .opacity(ping ? 0 : 0.85)
            // solid crosshair + ring
            Circle().stroke(.red, lineWidth: 4).frame(width: 50, height: 50)
            Rectangle().fill(.red).frame(width: 3, height: 40)
            Rectangle().fill(.red).frame(width: 40, height: 3)
        }
        .shadow(color: .black.opacity(0.6), radius: 2)
        .scaleEffect(landed ? 1 : 0.2)
        .opacity(landed ? 1 : 0)
        .onAppear {
            withAnimation(.spring(response: 0.45, dampingFraction: 0.5)) { landed = true }
            withAnimation(.easeOut(duration: 1.2).repeatForever(autoreverses: false)) { ping = true }
        }
    }
}

struct Turn: Identifiable {
    let id = UUID()
    let role: String      // "user" | "model"
    var text: String
    var stats: String = ""
    var image: UIImage? = nil
    var point: CGPoint? = nil   // normalized 0..1 grounding marker (Holo2 GUI grounding)
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
    @State private var showVLA = false           // BitVLA (image+instruction -> 7-DoF) sheet

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
                case .fastcontext: engine.mode = .fastcontext
                case .qwen3vl: engine.mode = .qwen3vl
                case .qwen3vl4b: engine.mode = .qwen3vl4b
                case .holo2vl: engine.mode = .holo2vl
                case .gemma4vl: engine.mode = .gemma4vl
                case .rwkv7: engine.mode = .rwkv7
                case .qwen3spec: engine.mode = .qwen3spec
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
        .sheet(isPresented: $showVLA) { BitVLAView() }
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
        // Two rows: a title/model/status row that never compresses (fixedSize keeps the title and
        // the .menu picker label on one line — a squeezed .menu label wraps a char per line), and a
        // secondary full-width segmented row for the modes that have a sub-choice. Keeping the wide
        // segmented control out of the top row is what stops the overflow.
        VStack(spacing: 6) {
            HStack(spacing: 10) {
                Text("CoreAIChat").font(.headline).fixedSize()
                // Model menu (scales as the zoo grows); fixedSize so its label never wraps.
                Picker("Model", selection: modelSelection) {
                    ForEach(ChatModel.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.menu)
                .fixedSize()
                .disabled(engine.busy || engine.loading)
                Spacer(minLength: 8)
                Text(engine.status)
                    .font(.caption).foregroundStyle(engine.ready ? .green : .secondary)
                    .lineLimit(1).truncationMode(.tail)
                // Free this model's files from the device (re-downloadable). Only when installed.
                if engine.installedForCurrentMode {
                    Button(role: .destructive) { showDeleteConfirm = true } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(engine.busy || engine.loading)
                }
                // BitVLA: zoo's first on-device VLA (image+instruction -> 7-DoF action).
                Button { showVLA = true } label: { Image(systemName: "hand.raised.fingers.spread") }
            }
            // Secondary control: the engine (gemma GPU/ANE/⚡), or the spec-decode ⚡Spec/Greedy
            // toggle (lossless — same output, only tok/s changes). Full width, own row.
            if engine.mode.chatModel == .gemma {
                Picker("Compute", selection: unitSelection) {
                    Text("GPU").tag(GemmaMode.gpu)
                    Text("ANE").tag(GemmaMode.ane)
                    Text("⚡").tag(GemmaMode.gemmaTbl)
                }
                .pickerStyle(.segmented)
                .disabled(engine.busy || engine.loading)
            } else if engine.mode == .qwen3spec {
                Picker("Spec", selection: $engine.specDecodeOn) {
                    Text("⚡ Spec").tag(true)
                    Text("Greedy").tag(false)
                }
                .pickerStyle(.segmented)
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
                    // Grounded result images (with a marker) render larger for clarity.
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: turn.point != nil ? 560 : 340)
                        .overlay {
                            if let p = turn.point {
                                GeometryReader { geo in
                                    GroundMarker()
                                        .position(x: p.x * geo.size.width, y: p.y * geo.size.height)
                                }
                            }
                        }
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
        var clean = engine.output
        if let r = clean.range(of: "(?s)<think>.*?</think>", options: .regularExpression) {
            clean.removeSubrange(r)
            clean = clean.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var t = Turn(role: "model", text: clean, stats: engine.stats)
        // Holo2 / VL grounding: if the reply carries a click point and an image is
        // attached, show the screenshot with a marker on the grounded element.
        if engine.mode.isVL, let img = attachedThumb, let pt = Self.parseHoloPoint(engine.output) {
            t.image = img
            t.point = pt
        }
        turns.append(t)
    }

    /// Parse Holo2's grounding output into a 0..1 fraction for overlaying a marker.
    /// Coords are normalized 0..1000; the model emits them in varied framing depending on the
    /// prompt: `<point x=471 y=507>`, `{374, 375}`, `(x, y)`, `[x, y]`. We strip the <think>
    /// block, prefer an explicit coordinate group, then take the first two numbers.
    static func parseHoloPoint(_ s: String) -> CGPoint? {
        var t = s
        if let r = t.range(of: "(?s)<think>.*?</think>", options: .regularExpression) {
            t.removeSubrange(r)
        }
        var scope = Substring(t)
        for p in ["<point[^>]*>", "\\{[^}]*\\}", "\\([^)]*\\)", "\\[[^\\]]*\\]"] {
            if let r = t.range(of: p, options: .regularExpression) { scope = t[r]; break }
        }
        var nums: [Double] = []
        var rest = scope
        while let r = rest.range(of: "-?[0-9]+(?:\\.[0-9]+)?", options: .regularExpression) {
            if let v = Double(rest[r]) { nums.append(v) }
            rest = rest[r.upperBound...]
            if nums.count >= 2 { break }
        }
        guard nums.count >= 2 else { return nil }
        func frac(_ v: Double) -> Double { let f = v > 1.0 ? v / 1000.0 : v; return min(max(f, 0), 1) }
        return CGPoint(x: frac(nums[0]), y: frac(nums[1]))
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
                    case .fastcontext: "~3.0"
                    case .qwen3vl: "~3.1"
                    case .qwen3vl4b: "~5.5"
                    case .holo2vl: "~5.2"
                    case .gemma4vl: "~4.7"
                    case .rwkv7: "~2.0"
                    case .qwen3spec: "~2.1"
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
