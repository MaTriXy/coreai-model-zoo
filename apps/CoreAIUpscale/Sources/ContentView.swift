// ContentView.swift — pick a photo from the photo library, run AdcSR ×4 on-device, compare
// input vs result. Cross-platform (iOS + macOS): PhotosPicker + ImageIO load the CGImage and
// SwiftUI renders it directly, so there is no UIKit/AppKit branching.

import CoreAIKitVision
import CoreGraphics
import ImageIO
import PhotosUI
import SwiftUI

@MainActor
final class UpscaleModel: ObservableObject {
    @Published var status = "Choose an image to upscale ×4"
    @Published var downloadFraction: Double?
    @Published var busy = false
    @Published var input: CGImage?
    @Published var output: CGImage?

    private var resolver: SuperResolver?

    func load(data: Data) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            status = "Could not read that photo"
            return
        }
        input = img
        output = nil
        status = "Loaded \(img.width)×\(img.height)"
    }

    func upscale() async {
        guard let input, !busy else { return }
        busy = true
        defer { busy = false }
        do {
            if resolver == nil {
                status = "Downloading AdcSR (~1.7 GB)…"
                resolver = try await SuperResolver(model: .adcsrX4) { [weak self] p in
                    Task { @MainActor in self?.downloadFraction = p.fraction }
                }
                downloadFraction = nil
            }
            status = "Upscaling ×4 on-device…"
            let started = Date()
            let out = try await resolver!.upscale(input)
            output = out
            status = String(
                format: "Done — %d×%d in %.1fs", out.width, out.height,
                Date().timeIntervalSince(started))
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

struct ContentView: View {
    @StateObject private var model = UpscaleModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                HStack {
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Choose photo", systemImage: "photo")
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        Task { await model.upscale() }
                    } label: {
                        Label("Upscale ×4", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.input == nil || model.busy)
                }

                if let f = model.downloadFraction {
                    ProgressView(value: f) { Text("Downloading model… \(Int(f * 100))%") }
                } else if model.busy {
                    ProgressView()
                }
                Text(model.status).font(.footnote).foregroundStyle(.secondary)

                if let input = model.input {
                    imageCard("Input — \(input.width)×\(input.height)", input)
                }
                if let output = model.output {
                    imageCard("AdcSR ×4 — \(output.width)×\(output.height)", output)
                }
            }
            .padding()
            .frame(maxWidth: 700)
        }
        .onChange(of: pickerItem) { _, item in
            Task {
                guard let item,
                    let data = try? await item.loadTransferable(type: Data.self)
                else { return }
                model.load(data: data)
            }
        }
    }

    @ViewBuilder private func imageCard(_ title: String, _ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Image(decorative: image, scale: 1)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 340)
                .background(Color.gray.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}
