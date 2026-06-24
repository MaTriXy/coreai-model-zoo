// SegmentationEngine — downloads / loads a Core AI SAM3 segmenter bundle and runs
// text-prompt segmentation using Apple's official CoreAIImageSegmenter runtime
// (apple/coreai-models). SAM3 takes an image + a text prompt ("cat") and returns
// instance masks, boxes, and scores; this engine composites the masks back onto the
// source image for display.
//
// Model delivery uses the shared AppShared/ModelDownloader (range-chunked parallel
// download with cross-launch resume and atomic bundle placement) for the `.aimodel`
// directory bundle + the tokenizer, plus a direct GET for the tiny root metadata.json
// that the HF tree API can't enumerate.

import CoreAIImageSegmenter
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// `ImageSegmenter` is a non-Sendable value type, so calling its `async` methods directly
/// from the `@MainActor` engine trips Swift 6's region-isolation check ("sending risks a
/// data race"). This box wraps it: the box itself is `Sendable`, and its async methods run
/// the inference off the main actor without sending the non-Sendable value across a boundary.
/// Safe in practice — the engine serializes calls (one segmentation at a time).
private struct SegmenterBox: @unchecked Sendable {
    let segmenter: ImageSegmenter
    func warmup() async throws { try await segmenter.warmup() }
    func segment(image: CGImage, prompt: String, parameters: SegmentationParameters) async throws
        -> SegmentationResponse
    {
        try await segmenter.segment(image: image, prompt: prompt, parameters: parameters)
    }
}

@MainActor
final class SegmentationEngine: ObservableObject {

    /// A Core AI-converted SAM3 segmenter bundle published on the Hugging Face Hub.
    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String           // "org/name" on the Hub
        let bundleDirName: String    // local folder name under Documents/
        let title: String
        let aimodelName: String      // the `.aimodel` directory inside the repo
    }

    static let catalog: [ModelOption] = [
        ModelOption(
            repoId: "mlboydaisuke/sam3-CoreAI-official",
            bundleDirName: "sam3-CoreAI",
            title: "SAM 3 (text-prompt)",
            aimodelName: "sam3_float16.aimodel")
    ]

    enum Status: Equatable {
        case idle, downloading, loading, ready
        case segmenting
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading model…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .segmenting: return "Segmenting…"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .segmenting: return true
            default: return false
            }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var sourceImage: CGImage?     // the picked image
    @Published private(set) var resultImage: CGImage?     // image with mask overlay
    @Published private(set) var segmentCount = 0
    @Published private(set) var segmentSeconds: Double?
    @Published private(set) var modelTitle = ""

    /// Shared range-chunked downloader (atomic placement + cross-launch resume).
    let downloader = ModelDownloader()

    private var segmenter: SegmenterBox?
    private var work: Task<Void, Never>?
    /// All segments from the last run (sorted by score); the confidence filter selects from these
    /// without re-running inference.
    private var lastSegments: [Segment] = []
    private(set) var confidence: Float = 0.5

    var canSegment: Bool { if case .ready = status { return true }; if case .segmenting = status { return false }; return segmenter != nil }
    var isDownloadingOrLoading: Bool { if case .downloading = status { return true }; if case .loading = status { return true }; return false }

    // The directory bundle + tokenizer are enumerable via the HF tree API; metadata.json
    // at the repo root is fetched with a plain resolve GET.
    private func directoryItems(for option: ModelOption) -> [ModelDownloader.Item] {
        [
            .init(remote: option.aimodelName, local: option.aimodelName),
            .init(remote: "tokenizer", local: "tokenizer"),
        ]
    }
    private static let rootFiles = ["metadata.json"]

    // MARK: - Image input

    func setImage(_ cg: CGImage) {
        sourceImage = cg
        resultImage = nil
        segmentCount = 0
        segmentSeconds = nil
        lastSegments = []
        if case .error = status { status = segmenter != nil ? .ready : .idle }
    }

    // MARK: - Loading

    /// Download a converted bundle from the Hugging Face Hub (cached after the first
    /// run, resumable across launches) and load it.
    func loadFromHub(_ option: ModelOption) {
        work?.cancel()
        modelTitle = option.title
        status = .downloading
        Self.setIdleTimerDisabled(true)
        work = Task {
            defer { Self.setIdleTimerDisabled(false) }
            do {
                let dest = try Self.bundleDestination(for: option)
                await downloader.fetch(
                    repo: "https://huggingface.co/\(option.repoId)",
                    items: directoryItems(for: option), into: dest)
                try Task.checkCancellation()
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                try await Self.fetchRootFiles(repoId: option.repoId, into: dest)
                try await loadBundle(at: dest)
            } catch is CancellationError {
                // user cancelled — cancel() owns the resulting state
            } catch {
                status = .error("\(error)")
            }
        }
    }

    /// Load a bundle directory the user picked locally.
    func loadLocal(_ url: URL) {
        work?.cancel()
        modelTitle = url.lastPathComponent
        work = Task {
            do { try await loadBundle(at: url) }
            catch { status = .error("\(error)") }
        }
    }

    private func loadBundle(at dir: URL) async throws {
        status = .loading
        let accessed = dir.startAccessingSecurityScopedResource()
        defer { if accessed { dir.stopAccessingSecurityScopedResource() } }
        let box = SegmenterBox(segmenter: try await ImageSegmenter(resourcesAt: dir.path))
        try await box.warmup()
        try Task.checkCancellation()
        segmenter = box
        status = .ready
    }

    // MARK: - Segmentation

    func segment(prompt: String, confidence: Float) {
        guard let segmenter, let source = sourceImage else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        self.confidence = confidence
        work?.cancel()
        status = .segmenting
        resultImage = nil
        work = Task {
            do {
                // Over-fetch candidates at a fixed pixel threshold; the confidence (score) filter
                // below selects which to show — and can be re-applied live without re-inference.
                let params = SegmentationParameters(maskThreshold: 0.5, maxSegments: 12)
                let start = DispatchTime.now().uptimeNanoseconds
                let response = try await segmenter.segment(image: source, prompt: text, parameters: params)
                try Task.checkCancellation()
                segmentSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                lastSegments = response.segments
                applyConfidence(self.confidence)
                status = .ready
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    /// Re-filter the last run's segments by score and re-render — cheap, no re-inference, so the
    /// confidence slider updates the overlay live.
    func applyConfidence(_ c: Float) {
        confidence = c
        guard let source = sourceImage, !lastSegments.isEmpty else { return }
        let kept = lastSegments.filter { $0.score >= c }
        resultImage = Self.renderOverlay(base: source, segments: kept)
        segmentCount = kept.count
    }

    func cancel() {
        // Cancelling the Task propagates structured-concurrency cancellation into the in-flight
        // `await downloader.fetch(...)`; the downloader has no separate cancel handle.
        work?.cancel()
        if case .downloading = status { status = segmenter != nil ? .ready : .idle }
        else if case .loading = status { status = segmenter != nil ? .ready : .idle }
        else if case .segmenting = status { status = .ready }
        Self.setIdleTimerDisabled(false)
    }

    // MARK: - Overlay rendering

    /// Composite each segment's binary mask onto the base image as a translucent colored
    /// fill, plus its bounding box and score label. Mask geometry is row-major at the
    /// input-image resolution; box origin differs by platform (see `Segment.box`).
    nonisolated static func renderOverlay(base: CGImage, segments: [Segment]) -> CGImage {
        let w = base.width, h = base.height
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return base }

        // CGContext is bottom-left origin. Draw the base via the standard fill rect.
        ctx.draw(base, in: CGRect(x: 0, y: 0, width: w, height: h))

        let palette: [(CGFloat, CGFloat, CGFloat)] = [
            (1.00, 0.23, 0.19), (0.20, 0.78, 0.35), (0.04, 0.52, 1.00),
            (1.00, 0.58, 0.00), (0.69, 0.32, 0.87), (0.00, 0.78, 0.75),
            (1.00, 0.18, 0.57), (0.55, 0.76, 0.29),
        ]

        for (i, seg) in segments.enumerated() where !seg.mask.isEmpty {
            let (r, g, b) = palette[i % palette.count]
            // Render the mask into its own RGBA buffer (foreground = translucent color), then
            // blit. The mask is row-major top-down; `CGContext.draw` renders a top-down CGImage
            // right-side-up (same as the base image), so keep the rows in source order — flipping
            // them here would double-flip and the mask would come out upside-down.
            let mw = seg.maskWidth, mh = seg.maskHeight
            var pixels = [UInt8](repeating: 0, count: mw * mh * 4)
            let cr = UInt8(r * 255), cg2 = UInt8(g * 255), cb = UInt8(b * 255)
            let alpha: UInt8 = 130
            for y in 0..<mh {
                let srcRow = y * mw
                let dstRow = y * mw
                for x in 0..<mw where seg.mask[srcRow + x] {
                    let p = (dstRow + x) * 4
                    // premultiplied-last: store color * alpha/255
                    pixels[p] = UInt8(Int(cr) * Int(alpha) / 255)
                    pixels[p + 1] = UInt8(Int(cg2) * Int(alpha) / 255)
                    pixels[p + 2] = UInt8(Int(cb) * Int(alpha) / 255)
                    pixels[p + 3] = alpha
                }
            }
            pixels.withUnsafeMutableBytes { raw in
                if let mctx = CGContext(
                    data: raw.baseAddress, width: mw, height: mh, bitsPerComponent: 8,
                    bytesPerRow: mw * 4, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                   let maskImg = mctx.makeImage() {
                    ctx.draw(maskImg, in: CGRect(x: 0, y: 0, width: w, height: h))
                }
            }

            // Bounding box outline.
            ctx.setStrokeColor(red: r, green: g, blue: b, alpha: 1.0)
            ctx.setLineWidth(max(2, CGFloat(w) / 320))
            ctx.stroke(boxInBottomUp(seg.box, imageHeight: h))
        }

        return ctx.makeImage() ?? base
    }

    /// SAM3 boxes come in pixel coordinates; on iOS the origin is top-left, on macOS
    /// bottom-left. Normalize to the bottom-up CGContext used for compositing.
    private nonisolated static func boxInBottomUp(_ box: CGRect, imageHeight h: Int) -> CGRect {
        #if canImport(UIKit)
        // UIKit: box origin is top-left → flip y for the bottom-up context.
        return CGRect(x: box.minX, y: CGFloat(h) - box.maxY, width: box.width, height: box.height)
        #else
        // AppKit: box is already bottom-left.
        return box
        #endif
    }

    // MARK: - Storage / platform shims

    private static func bundleDestination(for option: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(option.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Fetch the tiny root-level metadata.json the segmenter bundle needs alongside the
    /// `.aimodel` directory. The HF tree API only enumerates directories, so this is a
    /// plain resolve GET.
    private static func fetchRootFiles(repoId: String, into dest: URL) async throws {
        for name in rootFiles {
            let out = dest.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: out.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(name)") else { continue }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                throw err("failed to fetch \(name) (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1))")
            }
            try data.write(to: out, options: .atomic)
        }
    }

    private static func err(_ msg: String) -> NSError {
        NSError(domain: "SegmentationEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private static func setIdleTimerDisabled(_ disabled: Bool) {
        #if canImport(UIKit)
        Task { @MainActor in UIApplication.shared.isIdleTimerDisabled = disabled }
        #endif
    }
}
