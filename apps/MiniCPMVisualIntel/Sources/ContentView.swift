// ContentView.swift — coach + mirror. The value of this app lives in the OS: MiniCPM-V-4.6 answers
// the system Visual Intelligence "ask" on-device, with the app closed. So the app (1) downloads +
// warms the model (the background VI launch can only LOAD what the foreground already cached — it
// has no bandwidth to download 2 GB), (2) coaches you to trigger Visual Intelligence, (3) mirrors
// the exact engine path in a "Try it" panel so you can verify it on the phone, and (4) exposes the
// experimental "also answer inside Visual Intelligence" flag.

import PhotosUI
import SwiftUI
import UIKit

struct ContentView: View {
    @Environment(AppRouter.self) private var router
    @StateObject private var vm = MirrorVM()
    @StateObject private var downloader = ModelDownloader()
    @State private var pick: PhotosPickerItem?
    @State private var downloaded = MiniCPMVIEngine.bundlesPresent
    @State private var runInVI = MiniCPMVIEngine.runInVisualIntelligence

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    identity
                    if downloaded {
                        coachCard
                        viToggle
                        tryItCard
                    } else {
                        downloadCard
                    }
                }
                .padding()
            }
            .navigationTitle("MiniCPM-V 4.6")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { if downloaded { await vm.preload() } }
        .onChange(of: pick) { _, item in
            guard let item else { return }
            Task { @MainActor in
                if let data = try? await item.loadTransferable(type: Data.self),
                    let img = UIImage(data: data)
                {
                    await vm.answer(image: img)
                }
            }
        }
        // A capture handed back from Visual Intelligence ("Continue in app" / a tapped teaser):
        // answer it here in the foreground with the full memory budget.
        .onChange(of: router.pendingImageData) { _, data in
            guard downloaded, let data, let img = UIImage(data: data) else { return }
            Task { @MainActor in await vm.answer(image: img) }
        }
        // An answer already produced inside the VI background launch (flag A): just show it.
        .onChange(of: router.pendingAnswer) { _, text in
            guard let text else { return }
            vm.show(answer: text)
        }
    }

    // MARK: - Sections

    private var identity: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Your own model inside Visual Intelligence")
                .font(.title2.bold())
            Text("MiniCPM-V 4.6 — running on your device")
                .font(.subheadline).foregroundStyle(.secondary)
            Label("on-device · offline · no cloud", systemImage: "bolt.shield")
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(.tint.opacity(0.15), in: Capsule())
            Text(
                "Apple's Visual Intelligence \"ask\" sends your photo to ChatGPT in the cloud. "
                    + "This answers the same surface with a model on your phone — even in airplane mode."
            )
            .font(.footnote).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var downloadCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Download the model").font(.headline)
            Text(
                "MiniCPM-V 4.6 (~2.1 GB: a SigLIP vision tower + a qwen3.5-hybrid decoder) downloads "
                    + "once from Hugging Face, then runs fully on-device. Needed before Visual "
                    + "Intelligence can answer with the app closed."
            )
            .font(.footnote).foregroundStyle(.secondary)

            if downloader.busy {
                ProgressView(value: downloader.fraction)
                Text(downloader.detail).font(.caption).foregroundStyle(.secondary)
            } else if case .failed(let msg) = downloader.phase {
                Text("Download failed: \(msg)").font(.caption).foregroundStyle(.red)
            }

            Button {
                Task { @MainActor in
                    await downloader.fetch(
                        repo: MiniCPMVIEngine.repo, items: MiniCPMVIEngine.items,
                        into: MiniCPMVIEngine.modelsDir)
                    if MiniCPMVIEngine.bundlesPresent {
                        downloaded = true
                        await vm.preload()
                    }
                }
            } label: {
                Label(
                    downloader.busy ? "Downloading…" : "Download MiniCPM-V (~2.1 GB)",
                    systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(downloader.busy)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }

    private var coachCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Use it from the camera or a screenshot")
                .font(.headline)
            trigger("camera.aperture", "iPhone 16/17", "Press and hold Camera Control")
            trigger("button.programmable", "Other iPhone", "Action button / Control Center")
            trigger("camera.viewfinder", "Any screenshot", "Side + Volume Up → Visual Intelligence")
            Text("Then MiniCPM-V's answer appears in the system results — with this app closed.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }

    private func trigger(_ icon: String, _ title: String, _ detail: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).font(.title3).frame(width: 30)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var viToggle: some View {
        Toggle(isOn: $runInVI) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Also answer inside Visual Intelligence").font(.subheadline.weight(.semibold))
                Text(
                    "Experimental: run MiniCPM-V in the background VI launch so the answer surfaces "
                        + "live. Off = a tap-to-ask teaser that answers here in the app."
                )
                .font(.caption).foregroundStyle(.secondary)
            }
        }
        .onChange(of: runInVI) { _, on in MiniCPMVIEngine.runInVisualIntelligence = on }
        .padding()
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }

    private var tryItCard: some View {
        // Read main-actor state into locals here (in the @MainActor View body) so the PhotosPicker
        // label — a Sendable closure — captures only Sendable values, not `vm`.
        let pickerTitle = vm.image == nil ? "Pick a photo to ask" : "Change photo"
        return VStack(alignment: .leading, spacing: 12) {
            Text("Try it — the same engine the system query runs")
                .font(.headline)
            Text(vm.status).font(.caption).foregroundStyle(.secondary)

            if let img = vm.image {
                Image(uiImage: img).resizable().scaledToFit()
                    .frame(maxHeight: 220)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            PhotosPicker(selection: $pick, matching: .images) {
                Label(pickerTitle, systemImage: "photo")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!vm.ready)

            if !vm.answer.isEmpty || vm.busy {
                Text(vm.answer.isEmpty ? "…" : vm.answer)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding()
                    .background(.tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
                if !vm.info.isEmpty {
                    Text(vm.info).font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }
}

/// Foreground mirror: drives the SAME `MiniCPMVIEngine` path the Visual Intelligence query uses.
@MainActor
final class MirrorVM: ObservableObject {
    @Published var status = "loading MiniCPM-V 4.6…"
    @Published var ready = false
    @Published var image: UIImage?
    @Published var answer = ""
    @Published var busy = false
    @Published var info = ""

    private let engine = MiniCPMVIEngine.shared

    func preload() async {
        do {
            _ = try await engine.ready { self.status = $0 }
            ready = true
            status = "ready — pick a photo, or trigger Visual Intelligence"
        } catch {
            status = "load error: \(error.localizedDescription)"
        }
    }

    func answer(image img: UIImage) async {
        guard let cg = img.cgImage, !busy else { return }
        image = img
        answer = ""
        info = ""
        busy = true
        let t0 = Date()
        do {
            let text = try await engine.caption(
                cgImage: cg, onStatus: { self.status = $0 }, onUpdate: { self.answer = $0 })
            answer = text
            info = String(format: "on-device · %.1fs · no cloud", -t0.timeIntervalSinceNow)
            status = "ready — pick another photo, or trigger Visual Intelligence"
        } catch {
            answer = "error: \(error.localizedDescription)"
        }
        busy = false
    }

    func show(answer text: String) {
        image = nil
        answer = text
        info = "answered inside Visual Intelligence · on-device"
        status = "this answer came from the system Visual Intelligence flow"
    }
}
