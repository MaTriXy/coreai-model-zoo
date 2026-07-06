// DiffusionEngine — downloads / loads a Core AI diffusion bundle and generates
// images using Apple's official CoreAIDiffusionPipeline runtime. The pipeline type
// (FLUX.2 / SD3 / SD) is auto-detected from metadata.json, mirroring the zoo's
// `diffusion-runner` reference tool, so any `coreai.diffusion.export` bundle drops in.
//
// The hosted catalog is macOS-only: FLUX.2 klein 4B's peak footprint exceeds the iOS
// per-process memory limit (measured ~0.4 GB over on a 12 GB iPhone 17 Pro — the text
// encoder is not released before the transformer runs). The iOS app still loads smaller
// diffusion bundles (e.g. Stable Diffusion) via "Local…".
//
// Model delivery uses the shared AppShared/ModelDownloader (range-chunked parallel
// download with cross-launch resume and atomic bundle placement) for the `.aimodel`
// directory bundles + the tokenizer, plus a couple of direct GETs for the handful of
// tiny root files (metadata.json, vae_bn_*.npy) that the HF tree API can't enumerate.

import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

/// Thread-safe cancel flag — readable from the pipeline's (possibly off-main)
/// progress callback without touching @MainActor state.
final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
}

@MainActor
final class DiffusionEngine: ObservableObject {

    /// A Core AI-converted diffusion bundle published on the Hugging Face Hub.
    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String           // "org/name" on the Hub
        let bundleDirName: String    // local folder name under Documents/
        let title: String
        let defaultSteps: Int
        let defaultGuidance: Float
        // The Hub repo also hosts in-context edit transformers, fetched on demand (not in the
        // base download) so text-to-image users don't pay for the ~4 GB edit weights up front.
        var hasEditAssets: Bool = false
    }

    // Hosted catalog — macOS only. FLUX.2 klein 4B overruns the iOS memory limit, so the
    // iOS app ships with an empty catalog and loads smaller bundles via "Local…".
    static let catalog: [ModelOption] = {
        #if os(macOS)
        return [
            ModelOption(
                repoId: "mlboydaisuke/FLUX.2-klein-4B-CoreAI",
                bundleDirName: "FLUX.2-klein-4B",
                title: "FLUX.2 klein 4B",
                defaultSteps: 4, defaultGuidance: 1.0,
                hasEditAssets: true)
        ]
        #else
        return []
        #endif
    }()

    enum Status: Equatable {
        case idle
        case downloading
        case loading
        case ready
        case generating(step: Int, total: Int)
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading model…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .generating(let s, let t): return "Generating… step \(s)/\(t)"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .generating: return true
            default: return false
            }
        }
    }

    @Published var status: Status = .idle
    @Published var image: CGImage?
    @Published var exportURL: URL?
    @Published var savedURL: URL?
    @Published var modelTitle: String = "—"
    @Published var loadSeconds: Double?
    @Published var generateSeconds: Double?
    @Published var imageSize: String = ""
    /// True once a pipeline with a VAE encoder is loaded — FLUX.2 bundles ship one, so
    /// the loaded model can run image-to-image (the UI reveals its mode switch on this).
    @Published var supportsImg2Img: Bool = false
    /// Native square side the loaded model generates at (e.g. 1024). Used to letterbox an
    /// image-to-image source into the square the runtime expects, then crop the result back.
    private var modelSide: Int = 1024
    /// Directory the loaded bundle was read from — used to locate the edit-sequence transformer.
    private var modelDir: URL?
    /// True when the loaded FLUX.2 bundle also ships an edit-sequence transformer (in-context edit).
    @Published var supportsEdit: Bool = false
    /// True when the bundle also ships the two-reference edit transformer (combine two images).
    @Published var supports2refEdit: Bool = false
    /// True when the loaded hosted model can fetch its edit transformers on demand (not yet local).
    @Published var canDownloadEditAssets: Bool = false
    private var loadedRepoId: String?
    private var loadedHasEditAssets = false

    /// Shared range-chunked downloader (atomic placement + cross-launch resume).
    let downloader = ModelDownloader()

    private var pipeline: (any DiffusionPipeline)?
    private var descriptor: PipelineDescriptor?
    private var work: Task<Void, Never>?
    private var cancelToken = CancellationToken()

    var canGenerate: Bool { if case .ready = status { return true }; return false }

    // Platform target: iOS runs the lighter 512 / half-VAE components; macOS runs
    // the full 1024 components. The HF bundle is universal — we fetch only the subset
    // this platform needs. The transformer / VAE bundles are resolved by NAME at load
    // (Transformer / Transformer_512 …), so downloading only the half set is enough.
    #if os(iOS)
    private static let fluxMode: DecodeResolution = .half
    private static let decodeResolution: DecodeResolution = .half
    private static let directoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer_512.aimodel", local: "Transformer_512.aimodel"),
        .init(remote: "TextEncoder.aimodel", local: "TextEncoder.aimodel"),
        .init(remote: "VAEDecoder_half.aimodel", local: "VAEDecoder_half.aimodel"),
        .init(remote: "VAEEncoder_half.aimodel", local: "VAEEncoder_half.aimodel"),
        .init(remote: "tokenizer", local: "tokenizer"),
    ]
    // Edit transformers — fetched on demand (not part of the base download).
    private static let editDirectoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer_edit_512.aimodel", local: "Transformer_edit_512.aimodel"),
        .init(remote: "Transformer_edit_2ref_512.aimodel", local: "Transformer_edit_2ref_512.aimodel"),
    ]
    #else
    private static let fluxMode: DecodeResolution = .auto
    private static let decodeResolution: DecodeResolution = .full
    private static let directoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer.aimodel", local: "Transformer.aimodel"),
        .init(remote: "TextEncoder.aimodel", local: "TextEncoder.aimodel"),
        .init(remote: "VAEDecoder.aimodel", local: "VAEDecoder.aimodel"),
        .init(remote: "VAEEncoder.aimodel", local: "VAEEncoder.aimodel"),
        .init(remote: "tokenizer", local: "tokenizer"),
    ]
    // Edit transformers — fetched on demand (not part of the base download).
    private static let editDirectoryItems: [ModelDownloader.Item] = [
        .init(remote: "Transformer_edit.aimodel", local: "Transformer_edit.aimodel"),
        .init(remote: "Transformer_edit_2ref.aimodel", local: "Transformer_edit_2ref.aimodel"),
    ]
    #endif

    // Tiny root-level files the pipeline needs alongside the bundles. The HF tree API
    // only enumerates directories, so these are fetched with a plain resolve GET.
    private static let rootFiles = ["metadata.json", "vae_bn_mean.npy", "vae_bn_var.npy"]

    // MARK: - Loading

    /// Download a converted bundle from the Hugging Face Hub (cached after the first
    /// run, resumable across launches) and load it.
    func loadFromHub(_ option: ModelOption) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil
        modelTitle = option.title
        loadedRepoId = option.repoId
        loadedHasEditAssets = option.hasEditAssets
        status = .downloading
        // Keep the screen awake for the multi-GB download: if the device auto-locks the app
        // gets suspended and the (foreground) URLSession transfer stalls. Re-enabled when the
        // whole flow finishes (see the defer below).
        Self.setIdleTimerDisabled(true)

        // The fine-grained download progress (fraction / byte detail) is read straight off
        // `downloader` by the view (it's an ObservableObject); this Task only sequences the
        // phases and surfaces a terminal error.
        work = Task {
            defer { Self.setIdleTimerDisabled(false) }
            do {
                let dest = try Self.bundleDestination(for: option)
                await downloader.fetch(
                    repo: "https://huggingface.co/\(option.repoId)",
                    items: Self.directoryItems, into: dest)
                try Task.checkCancellation()   // cancelled mid-download → clean exit, not .error
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                try await Self.fetchRootFiles(repoId: option.repoId, into: dest)
                try await self.loadPipeline(at: dest)
            } catch is CancellationError {
                // user cancelled or started another action — cancel() owns the resulting state
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    private static func setIdleTimerDisabled(_ disabled: Bool) {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = disabled
        #endif
    }

    /// Load a bundle already exported to a local folder.
    func loadLocal(_ url: URL) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil
        modelTitle = url.lastPathComponent
        loadedRepoId = nil
        loadedHasEditAssets = false
        status = .loading
        work = Task {
            do { try await self.loadPipeline(at: url) }
            catch { self.status = .error("\(error)") }
        }
    }

    /// Edit-transformer bundle names for this platform (full 1024 vs half 512), by reference count.
    #if os(iOS)
    private static let edit1RefName = "Transformer_edit_512.aimodel"
    private static let edit2RefName = "Transformer_edit_2ref_512.aimodel"
    #else
    private static let edit1RefName = "Transformer_edit.aimodel"
    private static let edit2RefName = "Transformer_edit_2ref.aimodel"
    #endif
    private static func editTransformerName(refCount: Int) -> String {
        refCount >= 2 ? edit2RefName : edit1RefName
    }

    private func loadPipeline(at url: URL) async throws {
        status = .loading
        supportsImg2Img = false
        supportsEdit = false
        supports2refEdit = false
        canDownloadEditAssets = false
        modelDir = url
        let start = ContinuousClock.now
        let desc = try PipelineDescriptor.resolve(at: url, config: .auto)

        let built: any DiffusionPipeline
        switch desc.type {
        case .some(.flux2):
            built = try await Flux2Pipeline(from: url, config: .auto, mode: Self.fluxMode)
        case .some(.stableDiffusion3):
            built = try await SD3Pipeline(from: url, config: .auto)
        default:
            built = try await StableDiffusionPipeline.load(from: url, config: .auto)
        }

        self.descriptor = desc
        self.pipeline = built
        self.loadSeconds = Self.seconds(since: start)
        let size = built.defaultImageSize
        self.imageSize = "\(size.width)×\(size.height)"
        self.modelSide = size.width
        self.supportsImg2Img = built.supportsImageToImage
        let fm = FileManager.default
        self.supportsEdit = (desc.type == .flux2)
            && fm.fileExists(atPath: url.appendingPathComponent(Self.edit1RefName).path)
        self.supports2refEdit = self.supportsEdit
            && fm.fileExists(atPath: url.appendingPathComponent(Self.edit2RefName).path)
        // Hosted FLUX bundles can fetch the edit transformers on demand if not already present.
        self.canDownloadEditAssets = self.loadedHasEditAssets && !self.supportsEdit
        self.status = .ready
    }

    /// Documents/<bundleDirName> — a clean, dedicated folder that holds the placed
    /// bundles + root files. (Kept separate from any HF cache so a partial transfer
    /// can never leave a half-bundle the runtime would choke on.)
    private static func bundleDestination(for option: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(option.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Fetch the tiny root files with a plain resolve GET (skipping any already present).
    private static func fetchRootFiles(repoId: String, into dest: URL) async throws {
        let fm = FileManager.default
        for name in rootFiles {
            let target = dest.appendingPathComponent(name)
            if fm.fileExists(atPath: target.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(name)") else {
                throw err("bad file url for \(name)")
            }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                throw err("\(name): HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            try data.write(to: target, options: .atomic)
        }
    }

    // MARK: - Generation

    /// Generate an image. Pass `startingImage` (with `strength` in 0…1) to run image-to-image:
    /// the runtime encodes it through the VAE encoder, blends in noise up to `strength`, and
    /// denoises from there — higher strength deviates further from the source. `nil` is txt2img.
    func generate(prompt: String, negativePrompt: String, steps: Int, guidance: Float, seed: UInt32,
                  startingImage: CGImage? = nil, strength: Float = 1.0) {
        guard let pipeline, let desc = descriptor, canGenerate else { return }
        work?.cancel()
        savedURL = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)

        let scheduler: SchedulerType =
            (desc.type == .flux2 || desc.type == .stableDiffusion3)
            ? .discreteFlow : .dpmSolverMultistep

        // Image-to-image: the model only generates a fixed square. Letterbox the source into
        // that square (no stretching) and remember its aspect ratio to crop the result back.
        let sourceSize: (w: Int, h: Int)? = startingImage.map { ($0.width, $0.height) }
        let squaredSource = startingImage.map { Self.fitToSquare($0, side: modelSide) }

        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: steps,
            guidanceScale: guidance,
            schedulerType: scheduler,
            startingImage: squaredSource,
            strength: strength,
            encoderScaleFactor: desc.encoderScaleFactor ?? 0.18215,
            decoderScaleFactor: desc.decoderScaleFactor ?? 0.18215,
            decoderShiftFactor: desc.decoderShiftFactor ?? 0.0,
            decodeResolution: Self.decodeResolution,
            lazyModelLoading: true
        )

        work = Task {
            do {
                let start = ContinuousClock.now
                let result = try await pipeline.generateImages(configuration: config) { @Sendable progress in
                    let s = progress.step, t = progress.totalSteps
                    Task { @MainActor in self.status = .generating(step: s, total: t) }
                    return !token.isCancelled
                }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                var cg = result.images.first
                // Crop the square result back to the source's aspect ratio (image-to-image only).
                if let out = cg, let s = sourceSize, s.w != s.h {
                    cg = Self.cropToAspect(out, aspectW: s.w, aspectH: s.h)
                }
                self.image = cg
                if let cg { self.imageSize = "\(cg.width)×\(cg.height)" }
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    /// Single-reference convenience.
    func edit(referenceImage: CGImage, instruction: String, steps: Int, seed: UInt32) {
        edit(referenceImages: [referenceImage], instruction: instruction, steps: steps, seed: seed)
    }

    /// FLUX.2 in-context edit: denoise a fresh output while the transformer attends to one or more
    /// clean reference images, so the instruction edits/combines content while keeping the subjects.
    /// Requires the matching edit-sequence transformer (Transformer_edit / _2ref) in the bundle.
    func edit(referenceImages: [CGImage], instruction: String, steps: Int, seed: UInt32) {
        guard let flux = pipeline as? Flux2Pipeline, let dir = modelDir, canGenerate,
              !referenceImages.isEmpty else { return }
        work?.cancel()
        savedURL = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)

        let editName = Self.editTransformerName(refCount: referenceImages.count)
        let editURL = dir.appendingPathComponent(editName)
        work = Task {
            do {
                guard FileManager.default.fileExists(atPath: editURL.path) else {
                    self.status = .error("Edit transformer missing (\(editName)).")
                    return
                }
                let editTransformer = CoreAIDiffusionModelFunction(modelURL: editURL)
                let start = ContinuousClock.now
                let result = try await flux.editImages(
                    referenceImages: referenceImages, instruction: instruction,
                    editTransformer: editTransformer, stepCount: steps, seed: seed
                ) { @Sendable progress in
                    let s = progress.step, t = progress.totalSteps
                    Task { @MainActor in self.status = .generating(step: s, total: t) }
                    return !token.isCancelled
                }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                let cg = result.images.first
                self.image = cg
                if let cg { self.imageSize = "\(cg.width)×\(cg.height)" }
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    /// Fetch the in-context edit transformers on demand (they're not in the base download, so
    /// text-to-image users don't pay for the ~4 GB edit weights). Recomputes edit support after.
    func downloadEditAssets() {
        guard let repoId = loadedRepoId, let dir = modelDir, !isDownloadingOrLoading else { return }
        work?.cancel()
        canDownloadEditAssets = false
        status = .downloading
        Self.setIdleTimerDisabled(true)
        work = Task {
            defer { Self.setIdleTimerDisabled(false) }
            do {
                await downloader.fetch(
                    repo: "https://huggingface.co/\(repoId)",
                    items: Self.editDirectoryItems, into: dir)
                try Task.checkCancellation()
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                let fm = FileManager.default
                self.supportsEdit = fm.fileExists(atPath: dir.appendingPathComponent(Self.edit1RefName).path)
                self.supports2refEdit = self.supportsEdit
                    && fm.fileExists(atPath: dir.appendingPathComponent(Self.edit2RefName).path)
                self.canDownloadEditAssets = self.loadedHasEditAssets && !self.supportsEdit
                self.status = .ready
            } catch is CancellationError {
                self.canDownloadEditAssets = self.loadedHasEditAssets
                self.status = .ready
            } catch {
                self.canDownloadEditAssets = self.loadedHasEditAssets
                self.status = .error("\(error)")
            }
        }
    }

    /// Cancel the in-flight work. Generation falls back to the loaded model (.ready); a
    /// cancelled download/load drops to .idle (no model yet).
    func cancel() {
        cancelToken.cancel()
        work?.cancel()
        Self.setIdleTimerDisabled(false)
        switch status {
        case .generating: status = .ready
        case .downloading, .loading: status = .idle
        default: break
        }
    }

    var isDownloadingOrLoading: Bool {
        switch status { case .downloading, .loading: return true; default: return false }
    }

    // MARK: - Saving

    /// Copy the just-generated PNG into the user's Downloads (macOS) / a share-ready temp
    /// (iOS) and return the destination. Returns nil if there's nothing to save.
    @discardableResult
    func saveImageToDownloads() -> URL? {
        guard let src = exportURL else { return nil }
        let fm = FileManager.default
        let name = "coreai-\(modelTitle.replacingOccurrences(of: " ", with: "-"))-\(Int(Date().timeIntervalSince1970)).png"
        #if os(macOS)
        let dir = (try? fm.url(for: .downloadsDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        #else
        let dir = (try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        #endif
        let target = dir.appendingPathComponent(name)
        do {
            if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
            try fm.copyItem(at: src, to: target)
            savedURL = target
            return target
        } catch { return nil }
    }

    // MARK: - Helpers

    /// Letterbox an image into `side`×`side` without stretching: an aspect-fill (cover) copy
    /// fills the square as background, the whole aspect-fit image is drawn centered on top. The
    /// model then edits a coherent full-bleed square; the padded margins are cropped off after.
    private static func fitToSquare(_ image: CGImage, side: Int) -> CGImage {
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: 4 * side, space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        ctx.interpolationQuality = .high
        let w = CGFloat(image.width), h = CGFloat(image.height), s = CGFloat(side)
        // Cover background (fills the square, cropping the overflow).
        let cover = max(s / w, s / h)
        let cw = w * cover, ch = h * cover
        ctx.draw(image, in: CGRect(x: (s - cw) / 2, y: (s - ch) / 2, width: cw, height: ch))
        // Fit foreground (whole image, centered).
        let fit = min(s / w, s / h)
        let fw = w * fit, fh = h * fit
        ctx.draw(image, in: CGRect(x: (s - fw) / 2, y: (s - fh) / 2, width: fw, height: fh))
        return ctx.makeImage() ?? image
    }

    /// Crop a square result back to the source aspect ratio (the centered aspect-fit region).
    private static func cropToAspect(_ square: CGImage, aspectW: Int, aspectH: Int) -> CGImage {
        let s = CGFloat(min(square.width, square.height))
        let fit = min(s / CGFloat(aspectW), s / CGFloat(aspectH))
        let fw = (CGFloat(aspectW) * fit).rounded()
        let fh = (CGFloat(aspectH) * fit).rounded()
        let x = ((CGFloat(square.width) - fw) / 2).rounded()
        let y = ((CGFloat(square.height) - fh) / 2).rounded()
        return square.cropping(to: CGRect(x: x, y: y, width: fw, height: fh)) ?? square
    }

    /// Write a CGImage to a temp PNG for sharing/saving via SwiftUI ShareLink.
    private static func writeTempPNG(_ cg: CGImage?) -> URL? {
        guard let cg else { return nil }
        let name = "coreai-image-\(UInt32.random(in: 0 ... .max)).png"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? url : nil
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "DiffusionEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
