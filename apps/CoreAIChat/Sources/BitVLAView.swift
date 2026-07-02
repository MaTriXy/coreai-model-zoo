// BitVLAView — interactive on-device VLA demo: pick an image + a robot instruction, run BitVLA
// (1.58-bit ternary, custom kernel on the iPhone GPU), see the predicted 7-DoF action. Presented
// as a sheet from ChatView's header. Uses BitVLABackend directly (no chat engine). Instructions
// are presets (the phone carries precomputed text embeds, not a tokenizer/embed-table).

import CoreGraphics
import PhotosUI
import SwiftUI
import UIKit

struct BitVLAView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var backend = BitVLABackend()
    @State private var status = "loading models…"
    @State private var loaded = false
    @State private var photoItem: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var presetIndex = 0
    @State private var result: BitVLABackend.Result?
    @State private var running = false
    @State private var errorText: String?

    private let dofNames = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("1.58-bit Vision-Language-Action on the iPhone GPU. Pick an image and a "
                         + "robot instruction; BitVLA predicts a 7-DoF end-effector action.")
                        .font(.footnote).foregroundStyle(.secondary)

                    PhotosPicker(selection: $photoItem, matching: .images) {
                        ZStack {
                            if let image {
                                Image(uiImage: image).resizable().scaledToFit()
                                    .frame(maxHeight: 260).clipShape(RoundedRectangle(cornerRadius: 14))
                            } else {
                                RoundedRectangle(cornerRadius: 14).fill(Color(.systemGray6))
                                    .frame(height: 160)
                                    .overlay { Label("Choose image", systemImage: "photo.badge.plus") }
                            }
                        }
                    }
                    Button("Use sample image") { image = loadSample() }
                        .font(.caption)

                    Picker("Instruction", selection: $presetIndex) {
                        ForEach(Array(backend.presetTexts.enumerated()), id: \.offset) { i, t in
                            Text(t).tag(i)
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(!loaded)

                    Button {
                        Task { await predict() }
                    } label: {
                        HStack {
                            if running { ProgressView().tint(.white) }
                            Text(running ? "predicting…" : "Predict action")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!loaded || running || image == nil)

                    Text(status).font(.caption).foregroundStyle(loaded ? .green : .secondary)
                    if let errorText {
                        Text(errorText).font(.caption).foregroundStyle(.red)
                    }

                    if let r = result {
                        resultCard(r)
                    }
                }
                .padding()
            }
            .navigationTitle("BitVLA ⚡ VLA")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
            .task { await loadBackend() }
            .onChange(of: photoItem) { _, item in
                Task {
                    if let item, let data = try? await item.loadTransferable(type: Data.self),
                       let ui = UIImage(data: data) { image = ui; result = nil }
                }
            }
        }
    }

    private func resultCard(_ r: BitVLABackend.Result) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Predicted action (7-DoF, bridge_orig)").font(.subheadline.bold())
            ForEach(Array(r.dof.enumerated()), id: \.offset) { i, v in
                HStack {
                    Text(i < dofNames.count ? dofNames[i] : "d\(i)")
                        .font(.system(.caption, design: .monospaced)).frame(width: 64, alignment: .leading)
                    GeometryReader { geo in
                        let w = geo.size.width
                        ZStack(alignment: .leading) {
                            Rectangle().fill(Color(.systemGray5)).frame(height: 8)
                            // gripper (last dim) is 0..1; others ~[-1,1] -> center bar
                            let isGrip = (i == r.dof.count - 1)
                            let frac = isGrip ? CGFloat(max(0, min(1, v)))
                                              : CGFloat(max(-1, min(1, v)) * 0.5 + 0.5)
                            Capsule().fill(Color.accentColor)
                                .frame(width: max(4, w * frac), height: 8)
                        }
                    }.frame(height: 8)
                    Text(String(format: "% .3f", v))
                        .font(.system(.caption, design: .monospaced)).frame(width: 64, alignment: .trailing)
                }
            }
            Text(String(format: "vision %.0f ms · prefill %.0f ms (%d tok) · decode %.0f ms",
                        r.visionMs, r.prefillMs, r.tokens.count, r.decodeMs))
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding().background(Color(.systemGray6)).clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func loadBackend() async {
        guard !loaded else { return }
        do {
            try await backend.load()
            loaded = true
            status = "ready — \(backend.presetTexts.count) presets"
            if image == nil { image = loadSample() }
        } catch {
            status = "load failed"; errorText = "\(error)"
        }
    }

    private func predict() async {
        guard let image, let cg = image.cgImage else { errorText = "no image"; return }
        running = true; errorText = nil; status = "predicting…"
        do {
            let r = try await backend.predict(cgImage: cg, presetIndex: presetIndex)
            result = r
            status = "done"
        } catch {
            status = "predict failed"; errorText = "\(error)"
        }
        running = false
    }

    private func loadSample() -> UIImage? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = docs.appendingPathComponent("models")
            .appendingPathComponent(BitVLABackend.dataDir).appendingPathComponent("sample.png")
        return UIImage(contentsOfFile: url.path)
    }
}
