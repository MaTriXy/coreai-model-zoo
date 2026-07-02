// ActionView — live camera + V-JEPA 2 action recognition overlay (top-3 with confidence bars).
import SwiftUI

@MainActor
final class ActionVM: ObservableObject {
    @Published var status = "Tap Start — loads the model and the camera."
    @Published var running = false
    @Published var predictions: [ActionPrediction] = []
    @Published var inferMS = 0.0

    let feed = CameraFeed()
    private var engine: ActionEngine?
    private var loop: Task<Void, Never>?

    func toggle() async {
        if running { stop(); return }
        if engine == nil {
            guard let root = VJEPAAssets.root else {
                status = "Model not found — stage the bundle at \(VJEPAAssets.location.path)"; return
            }
            status = "Loading V-JEPA 2 (675 MB)…"
            do { engine = try await ActionEngine(root: root) } catch { status = "Load failed: \(error)"; return }
        }
        status = "Starting camera…"
        guard await feed.start() else { status = "Camera unavailable (check permission)."; return }
        running = true
        status = "Watching…"
        let feed = self.feed, engine = self.engine!
        loop = Task { [weak self] in
            while !Task.isCancelled {
                guard let frames = feed.snapshot() else {
                    try? await Task.sleep(for: .milliseconds(200)); continue
                }
                let tensor = await Task.detached { ActionEngine.tensor(from: frames) }.value
                let t0 = ContinuousClock.now
                guard let preds = try? await engine.classify(tensor) else { break }
                let dt = ContinuousClock.now - t0
                await MainActor.run {
                    guard let self, self.running else { return }
                    self.predictions = preds
                    self.inferMS = Double(dt.components.seconds) * 1000 + Double(dt.components.attoseconds) / 1e15
                }
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    func stop() {
        loop?.cancel(); loop = nil
        feed.stop()
        running = false
        predictions = []
        status = "Stopped."
    }
}

struct ActionView: View {
    @StateObject private var vm = ActionVM()

    var body: some View {
        ZStack(alignment: .bottom) {
            if vm.running {
                CameraPreview(session: vm.feed.session).ignoresSafeArea()
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "video.badge.waveform").font(.system(size: 44)).foregroundStyle(.secondary)
                    Text("coreai-video — on-device action recognition").font(.title3).bold()
                    Text("V-JEPA 2 (Meta's video world model) on Core AI. Point the camera at an action — lifting, rolling, covering… — and it names it. Nothing leaves the device.")
                        .font(.callout).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center).frame(maxWidth: 420)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(alignment: .leading, spacing: 8) {
                if vm.running {
                    ForEach(vm.predictions) { p in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(p.label).font(.callout.weight(p.id == vm.predictions.first?.id ? .semibold : .regular))
                                .lineLimit(1)
                            GeometryReader { geo in
                                ZStack(alignment: .leading) {
                                    Capsule().fill(.quaternary)
                                    Capsule().fill(.tint).frame(width: geo.size.width * CGFloat(p.prob))
                                }
                            }.frame(height: 5)
                        }
                    }
                    if vm.inferMS > 0 {
                        Text(String(format: "%.0f ms / 16-frame clip · on-device", vm.inferMS))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Button { Task { await vm.toggle() } } label: {
                        Label(vm.running ? "Stop" : "Start", systemImage: vm.running ? "stop.fill" : "video.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                Text(vm.status).font(.footnote).foregroundStyle(.secondary)
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding(12)
        }
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 480)
        #endif
    }
}
