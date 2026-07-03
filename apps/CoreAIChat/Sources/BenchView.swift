// BenchView — the Bench tab: pick a cataloged model, Run, get a result card and a
// shareable JSON blob. The blob (not a screenshot, not a typed number) is the submission
// unit: paste it into the zoo's bench-result issue form and your device row appears in
// BENCHMARKS.md after aggregation. See BenchRunner for the protocol and trust model.

import SwiftUI

struct BenchView: View {
    @ObservedObject var chatEngine: Gemma4ChatEngine
    @StateObject private var runner = BenchRunner()
    @StateObject private var downloader = ModelDownloader()
    @State private var selectedID = BenchModel.catalog[0].id
    @State private var copied = false
    @Environment(\.openURL) private var openURL

    private var selected: BenchModel {
        BenchModel.catalog.first { $0.id == selectedID } ?? BenchModel.catalog[0]
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    modelPanel
                    if let blob = runner.blob { resultCard(blob) }
                    if !runner.lines.isEmpty { logPanel }
                    Color.clear.frame(height: 1).id("end")
                }
                .padding()
            }
            .onChange(of: runner.lines.count) { proxy.scrollTo("end", anchor: .bottom) }
        }
        .safeAreaInset(edge: .top) { header }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("Community Bench").font(.headline).fixedSize()
            Spacer()
            Text(runner.running ? "running…" : "field data · fixed protocol")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .glassEffect()
        .padding(.horizontal, 8)
    }

    // MARK: model selection + run

    private var modelPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Your device measures a fixed protocol (128-token prompt → 256 greedy tokens, 1 cold + 3 warm). Share the result blob on GitHub to add your device row to the community matrix.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Picker("Model", selection: $selectedID) {
                    ForEach(BenchModel.catalog) { m in
                        Text(m.label).tag(m.id)
                    }
                }
                .pickerStyle(.menu)
                .disabled(runner.running || downloader.busy)
                Spacer()
                if selected.isInstalled {
                    Label("installed", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                }
            }
            if selected.isInstalled {
                Button {
                    Task {
                        // Free the chat model first — the bench needs the memory headroom
                        // and an idle GPU. (CoreAIChatApp reloads chat lazily on tab return.)
                        chatEngine.unload()
                        await runner.run(model: selected)
                    }
                } label: {
                    Label(runner.running ? "Running…" : "Run benchmark",
                          systemImage: "speedometer")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(runner.running || downloader.busy)
            } else {
                downloadPanel
            }
        }
        .padding(12)
        .glassEffect(in: .rect(cornerRadius: 16))
    }

    private var downloadPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            if downloader.busy {
                ProgressView(value: downloader.fraction)
                Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Button {
                    Task {
                        await downloader.fetch(
                            repo: "https://huggingface.co/" + selected.repo,
                            items: [ModelDownloader.Item(remote: selected.remotePath,
                                                         local: selected.bundleName)],
                            into: FileManager.default.urls(for: .documentDirectory,
                                                           in: .userDomainMask)[0]
                                .appendingPathComponent("models"))
                    }
                } label: {
                    Label(String(format: "Download %@ (~%.1f GB)", selected.label,
                                 selected.approxDownloadGB),
                          systemImage: "arrow.down.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
            if case .failed(let msg) = downloader.phase {
                Text(msg).font(.caption).foregroundStyle(.red)
            }
        }
    }

    // MARK: result card

    private func resultCard(_ blob: BenchBlob) -> some View {
        let warm = blob.results.runs.filter { $0.kind == "warm" }.map(\.decode_tok_s).sorted()
        let warmMed = warm.isEmpty ? 0 : warm[warm.count / 2]
        let cold = blob.results.runs.first { $0.kind == "cold" }
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(blob.model.id).font(.subheadline).fontWeight(.semibold)
                    Text("\(blob.device.model_identifier) · \(blob.device.os)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f tok/s", warmMed))
                        .font(.title2).fontWeight(.bold).monospacedDigit()
                    Text("decode · median of \(warm.count) warm runs")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 3) {
                GridRow {
                    Text("load").foregroundStyle(.secondary)
                    Text(String(format: "%.1f s", blob.results.load_s)).monospacedDigit()
                }
                if let cold {
                    GridRow {
                        Text("cold run").foregroundStyle(.secondary)
                        Text(String(format: "prefill %.1f · decode %.1f tok/s",
                                    cold.prefill_tok_s, cold.decode_tok_s)).monospacedDigit()
                    }
                }
                GridRow {
                    Text("thermal").foregroundStyle(.secondary)
                    Text("\(blob.environment.thermal_state_before) → \(blob.environment.thermal_state_after)")
                }
            }
            .font(.caption)

            HStack(spacing: 8) {
                Button {
                    UIPasteboard.general.string = runner.blobJSON
                    copied = true
                    // Prefilled issue form when the blob fits in a URL; else the bare
                    // template — the blob is on the clipboard either way.
                    openURL(BenchRunner.submissionURL(blob: blob, blobJSON: runner.blobJSON)
                        ?? BenchRunner.templateURL)
                } label: {
                    Label("Submit on GitHub", systemImage: "arrow.up.forward.app")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                Button {
                    UIPasteboard.general.string = runner.blobJSON
                    copied = true
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)
                ShareLink(item: runner.blobJSON) {
                    Image(systemName: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
            }
            Text("Submitting opens a GitHub issue with this blob — your device becomes a row in BENCHMARKS.md. No account data leaves the device.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(12)
        .glassEffect(in: .rect(cornerRadius: 16))
    }

    // MARK: log

    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(runner.lines.enumerated()), id: \.offset) { _, line in
                Text(line)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .glassEffect(in: .rect(cornerRadius: 16))
    }
}
