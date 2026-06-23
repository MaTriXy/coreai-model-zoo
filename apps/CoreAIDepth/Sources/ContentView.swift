// ContentView.swift — Depth Anything 3 monocular depth on-device. Two modes:
//   • Photo  — pick an image from the library, run depth once, show input + colorized depth.
//   • Camera — live depth from the camera at a few FPS (iOS).
// Cross-platform (iOS + macOS) via PhotosPicker + CoreGraphics; no UIKit/AppKit branching.

import CoreAIKitVision
import CoreGraphics
import ImageIO
import PhotosUI
import SwiftUI

// MARK: - Colormap (DA3 convention: inverse-depth, percentile-normalized, Spectral — far = red, near = blue)

enum DepthColormap {
    // Spectral-like control points (t = 0 far → 1 near).
    private static let stops: [(Double, Double, Double)] = [
        (0.84, 0.24, 0.31), (0.99, 0.68, 0.38), (1.00, 1.00, 0.75),
        (0.40, 0.76, 0.65), (0.20, 0.34, 0.65),
    ]

    private static func ramp(_ t: Double) -> (UInt8, UInt8, UInt8) {
        let x = max(0, min(1, t)) * Double(stops.count - 1)
        let i = min(Int(x), stops.count - 2)
        let f = x - Double(i)
        let a = stops[i], b = stops[i + 1]
        let r = a.0 + (b.0 - a.0) * f
        let g = a.1 + (b.1 - a.1) * f
        let bl = a.2 + (b.2 - a.2) * f
        return (UInt8(r * 255), UInt8(g * 255), UInt8(bl * 255))
    }

    /// Colorized RGBA8 image from a relative-depth map.
    static func image(from map: DepthMap) -> CGImage? {
        let n = map.width * map.height
        guard n > 0, map.values.count >= n else { return nil }
        // inverse depth (disparity); larger = nearer
        var disp = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let v = map.values[i]
            disp[i] = (v.isFinite && v > 1e-6) ? 1.0 / Double(v) : 0
        }
        let sorted = disp.sorted()
        let lo = sorted[Int(0.02 * Double(n))]
        let hi = sorted[min(n - 1, Int(0.98 * Double(n)))]
        let range = max(hi - lo, 1e-9)

        var rgba = [UInt8](repeating: 255, count: n * 4)
        for i in 0..<n {
            let (r, g, b) = ramp((disp[i] - lo) / range)
            rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b
        }
        return rgba.withUnsafeMutableBytes { ptr -> CGImage? in
            guard let ctx = CGContext(
                data: ptr.baseAddress, width: map.width, height: map.height,
                bitsPerComponent: 8, bytesPerRow: map.width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return nil }
            return ctx.makeImage()
        }
    }
}

// MARK: - Photo mode

private func resized(_ img: CGImage, to w: Int, to h: Int) -> CGImage? {
    guard w > 0, h > 0,
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return img }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()
}

@MainActor
final class DepthPhotoModel: ObservableObject {
    @Published var status = "Choose a photo to estimate depth"
    @Published var downloadFraction: Double?
    @Published var busy = false
    @Published var input: CGImage?
    @Published var depth: CGImage?

    private var estimator: DepthEstimator?

    func load(data: Data) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            status = "Could not read that photo"
            return
        }
        input = img
        depth = nil
        status = "Loaded \(img.width)×\(img.height)"
    }

    func estimate() async {
        guard let input, !busy else { return }
        busy = true
        defer { busy = false }
        do {
            if estimator == nil {
                status = "Downloading Depth Anything 3 (~54 MB)…"
                estimator = try await DepthEstimator(model: .depthAnything3Small) { [weak self] p in
                    Task { @MainActor in self?.downloadFraction = p.fraction }
                }
                downloadFraction = nil
            }
            status = "Estimating depth on-device…"
            let t0 = Date()
            let map = try await estimator!.estimateDepth(for: input)
            // Render the square 504² depth back at the input's aspect ratio.
            depth = DepthColormap.image(from: map).flatMap {
                resized($0, to: input.width, to: input.height)
            }
            status = String(format: "Done in %.2fs", Date().timeIntervalSince(t0))
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Camera mode (reuses CoreAIKit's CameraFeed + DepthEstimator)

@MainActor
@Observable
final class DepthCameraModel {
    var cameraImage: CGImage?
    var depthImage: CGImage?
    var status = "Loading model…"
    var inferenceMS: Double?

    private var feed: CameraFeed?
    private var started = false

    func start() async {
        guard !started else { return }
        started = true
        do {
            let estimator = try await DepthEstimator(model: .depthAnything3Small) { p in
                Task { @MainActor in self.status = "Downloading… \(Int(p.fraction * 100))%" }
            }
            status = "Starting camera…"
            let feed = CameraFeed(framesPerSecond: 5)
            self.feed = feed
            for await frame in try await feed.start() {
                let t0 = SuspendingClock.now
                let map = try await estimator.estimateDepth(for: frame)
                let e = (SuspendingClock.now - t0).components
                inferenceMS = Double(e.seconds) * 1000 + Double(e.attoseconds) / 1e15
                cameraImage = frame
                depthImage = DepthColormap.image(from: map)
                status = "Live"
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    func stop() { feed?.stop() }
}

// MARK: - UI

enum Mode: String, CaseIterable { case photo = "Photo", camera = "Camera" }

struct ContentView: View {
    @State private var mode: Mode = .photo

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding()

            switch mode {
            case .photo: PhotoView()
            case .camera: CameraView()
            }
        }
    }
}

private struct PhotoView: View {
    @StateObject private var model = DepthPhotoModel()
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
                        Task { await model.estimate() }
                    } label: {
                        Label("Estimate depth", systemImage: "cube.transparent")
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

                if let input = model.input { imageCard("Input", input) }
                if let depth = model.depth { imageCard("Depth", depth) }
            }
            .padding()
            .frame(maxWidth: 700)
        }
        .onChange(of: pickerItem) { _, item in
            Task {
                guard let item, let data = try? await item.loadTransferable(type: Data.self)
                else { return }
                model.load(data: data)
            }
        }
    }

    @ViewBuilder private func imageCard(_ title: String, _ image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Image(decorative: image, scale: 1)
                .resizable().scaledToFit()
                .frame(maxHeight: 360)
                .background(Color.gray.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct CameraView: View {
    @State private var model = DepthCameraModel()

    var body: some View {
        VStack(spacing: 8) {
            frame(model.cameraImage, "Camera")
            frame(model.depthImage, "Depth")
            HStack {
                Text(model.status)
                if let ms = model.inferenceMS { Text(String(format: "· %.0f ms/frame", ms)) }
                Spacer()
            }
            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            .padding(.horizontal, 12)
            Spacer()
        }
        .padding(.vertical, 8)
        .task { await model.start() }
        .onDisappear { model.stop() }
    }

    private func frame(_ image: CGImage?, _ label: String) -> some View {
        ZStack(alignment: .topLeading) {
            if let image {
                Image(decorative: image, scale: 1).resizable().scaledToFit()
            } else {
                Rectangle().fill(Color.gray.opacity(0.15)).aspectRatio(4 / 3, contentMode: .fit)
            }
            Text(label).font(.caption2).padding(4)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6)).padding(6)
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 8)
    }
}
