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
import CoreAIShared
import CoreGraphics
import Foundation
import ImageIO
import Tokenizers
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
        // GLM-Image (AR+diffusion hybrid) uses the bespoke GlmImagePipeline, not the high-level
        // diffusion pipeline. Its bundle is a folder of AR/DiT/VAE .aimodelc + tokenizer + ehs.f32.
        // One Hub repo hosts both resolutions (the AR is shared); glmSize picks the DiT/VAE pair.
        var isGLM: Bool = false
        var glmSize: Int = 1024
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
                hasEditAssets: true),
            // GLM-Image (zai-org, MIT) — AR + flow-matching diffusion hybrid. 1024 is native
            // quality; 512 is a faster variant (same weights, smaller static graph). Both live in
            // ONE Hub repo (the 9.6 GB AR is shared); "Download & Load" loads a bundle already
            // present under Documents/ or fetches it from the Hub otherwise.
            ModelOption(
                repoId: "mlboydaisuke/GLM-Image-CoreAI",
                bundleDirName: "GLM-Image-1024",
                title: "GLM-Image 1024 (AR+diffusion)",
                defaultSteps: 20, defaultGuidance: 1.5,
                isGLM: true, glmSize: 1024),
            ModelOption(
                repoId: "mlboydaisuke/GLM-Image-CoreAI",
                bundleDirName: "GLM-Image-512",
                title: "GLM-Image 512 (AR+diffusion)",
                defaultSteps: 20, defaultGuidance: 1.5,
                isGLM: true, glmSize: 512),
        ]
        #else
        return []
        #endif
    }()

    // GLM bundle contents on the Hub. The repo hosts both resolutions; the size-specific
    // DiT/VAE download to size-agnostic local names so the pipeline's resolver finds them.
    private static func glmItems(size: Int) -> [ModelDownloader.Item] {
        [
            .init(remote: "glm_image_ar.aimodelc", local: "glm_image_ar.aimodelc"),
            .init(remote: "glm_image_dit_\(size).aimodelc", local: "glm_image_dit.aimodelc"),
            .init(remote: "glm_image_vae_\(size).aimodel", local: "glm_image_vae.aimodel"),
            .init(remote: "tokenizer", local: "tokenizer"),
        ]
    }
    private static let glmRootFiles = ["ehs.f32"]

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

    /// GLM-Image (AR+diffusion hybrid) runs a bespoke low-level pipeline (GlmImagePipeline) instead
    /// of the high-level CoreAIDiffusionPipeline auto-detect path. Set when such a bundle is loaded.
    @Published var isGLM = false
    private var glm: GlmImagePipeline?

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
                let items = option.isGLM ? Self.glmItems(size: option.glmSize) : Self.directoryItems
                let roots = option.isGLM ? Self.glmRootFiles : Self.rootFiles
                // Load a bundle already present under Documents/ without re-downloading; otherwise
                // fetch it from the Hub. (GLM: the folder is "complete" once the AR bundle + ehs.f32
                // are present; FLUX always goes through the downloader, which no-ops cached files.)
                if !(option.isGLM && Self.glmBundleComplete(at: dest)) {
                    await downloader.fetch(
                        repo: "https://huggingface.co/\(option.repoId)",
                        items: items, into: dest)
                    try Task.checkCancellation()   // cancelled mid-download → clean exit, not .error
                    if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                    try await Self.fetchRootFiles(repoId: option.repoId, names: roots, into: dest)
                }
                try await self.loadPipeline(at: dest)
            } catch is CancellationError {
                // user cancelled or started another action — cancel() owns the resulting state
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    /// True when a GLM bundle folder already holds everything the pipeline needs locally.
    private static func glmBundleComplete(at dir: URL) -> Bool {
        GlmImagePipeline.looksLikeGLM(dir)
            && FileManager.default.fileExists(atPath: dir.appendingPathComponent("ehs.f32").path)
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

        // GLM-Image bundle (AR+diffusion hybrid) — bespoke low-level pipeline, not the auto-detect path.
        if GlmImagePipeline.looksLikeGLM(url) {
            let glmStart = ContinuousClock.now
            let pipe = GlmImagePipeline(dir: url)
            try await pipe.load()
            self.glm = pipe
            self.pipeline = nil
            self.descriptor = nil
            self.isGLM = true
            self.loadSeconds = Self.seconds(since: glmStart)
            self.imageSize = "\(pipe.side)×\(pipe.side)"
            self.modelSide = pipe.side
            self.status = .ready
            return
        }
        self.glm = nil
        self.isGLM = false

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
    private static func fetchRootFiles(repoId: String, names: [String] = rootFiles, into dest: URL) async throws {
        let fm = FileManager.default
        for name in names {
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
        if isGLM { generateGLM(prompt: prompt, steps: steps, guidance: guidance, seed: seed); return }
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

    /// GLM-Image generation (text→image only). Drives the bespoke AR→DiT→VAE pipeline. The UI's
    /// Steps (default 20 — quality reference; ~12 is a faster preview) and Guidance (1.5) are
    /// honored via a dynamically computed flow-match schedule.
    private func generateGLM(prompt: String, steps: Int, guidance: Float, seed: UInt32) {
        guard let glm, canGenerate else { return }
        work?.cancel()
        savedURL = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)
        work = Task {
            do {
                let start = ContinuousClock.now
                let cg = try await glm.generate(prompt: prompt, seed: UInt64(seed),
                                                steps: steps, guidance: guidance) { @Sendable step, total in
                    Task { @MainActor in self.status = .generating(step: step, total: total) }
                    return !token.isCancelled
                }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                self.image = cg
                self.imageSize = "\(cg.width)×\(cg.height)"
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch is CancellationError {
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

// MARK: - GLM-Image (AR + diffusion hybrid) pipeline

/// GLM-Image (zai-org, MIT, 16B) can't use the high-level CoreAIDiffusionPipeline auto-detect path:
/// it's an autoregressive GLM-4-9B visual-token generator feeding a flow-matching DiT. This drives
/// the three Core AI bundles + a host loop directly — a faithful port of ondevice/GlmImageRunner,
/// which was verified numerically exact vs the Python reference (PSNR 69 dB, pixel-identical with
/// matched prior+noise) and byte-exact on tokenization.
///
/// A GLM bundle folder holds:  glm_image_ar…aimodelc · glm_image_dit…aimodelc · glm_image_vae…aimodel
///                             · a tokenizer dir (tokenizer.json) · ehs.f32 (empty-glyph text embed [1472])
@MainActor
final class GlmImagePipeline {
    // AR / mrope
    private let nLayers = 40, nKV = 2, headDim = 128, kvSeq = 2048, rot = 64, visionVocab = 16512
    private let mrope = [8, 12, 12]
    private let theta = 10000.0
    private lazy var inv: [Double] = (0..<rot / 2).map { 1.0 / pow(theta, Double(2 * $0) / Double(rot)) }

    // Per-size config (glyph-free T2I). GLM-Image's 512 subset vs its native 1024, selected at
    // load time from the DiT graph's latent shape. 1024 emits a 32×32 large prior grid
    // (→ 2×-upsampled 64×64 = 4096 tokens, 4× the 512 path) at 2× spatial resolution = much
    // sharper. Suffix = <sop>H/32 W/32<eop> coarse + <sop>16 16<eop> fine + IMG_START; verified
    // byte-exact vs the HF GlmImageProcessor.
    struct SizeCfg { let side: Int; let suffix: [Int]; let grids: [(Int, Int)] }
    static let cfg512 = SizeCfg(
        side: 512,
        suffix: [167845, 115829, 16732, 115829, 167846, 167845, 115829, 16732, 115829, 167846, 16384],
        grids: [(16, 16), (16, 16)])
    static let cfg1024 = SizeCfg(
        side: 1024,
        suffix: [167845, 117687, 16732, 117687, 167846, 167845, 115829, 16732, 115829, 167846, 16384],
        grids: [(32, 32), (16, 16)])

    /// Flow-match schedule for any step count (verified vs the diffusers scheduler, max diff 5e-7):
    /// raw integer timesteps trunc(linspace(1000,1,steps+1)) condition the DiT (adaLN — must stay
    /// UNSHIFTED); the latent euler steps use the mu-shifted sigmas σ' = μ/(μ + (1/σ − 1)) with the
    /// resolution shift μ = 0.75·side/256 + 0.25 (1.75 @512, 3.25 @1024).
    static func schedule(side: Int, steps: Int) -> (rawTs: [Float], sigmas: [Float]) {
        let mu = 0.75 * Double(side) / 256.0 + 0.25
        var rawTs: [Float] = []
        var sigmas: [Float] = []
        for i in 0..<steps {
            let t = Double(Int(1000.0 - 999.0 * Double(i) / Double(steps)))   // trunc(linspace)
            rawTs.append(Float(t))
            sigmas.append(Float(mu / (mu + (1000.0 / t - 1.0))))
        }
        sigmas.append(0)
        return (rawTs, sigmas)
    }

    private let dir: URL
    private var arFn: InferenceFunction?, arD: InferenceFunctionDescriptor?
    private var ditFn: InferenceFunction?, ditD: InferenceFunctionDescriptor?
    private var vaeFn: InferenceFunction?, vaeD: InferenceFunctionDescriptor?
    private var tokenizer: (any Tokenizer)?
    private var ehs: [Float] = []
    private var cfg = GlmImagePipeline.cfg512   // replaced from the loaded DiT's latent shape in load()
    var side: Int { cfg.side }

    init(dir: URL) { self.dir = dir }

    /// True if `url` looks like a GLM-Image bundle (has an `…ar….aimodelc`).
    static func looksLikeGLM(_ url: URL) -> Bool { resolve(in: url, contains: "ar", ext: "aimodelc") != nil }

    // MARK: Loading

    func load() async throws {
        guard let arURL = Self.resolve(in: dir, contains: "ar", ext: "aimodelc"),
              let ditURL = Self.resolve(in: dir, contains: "dit", ext: "aimodelc"),
              let vaeURL = Self.resolve(in: dir, contains: "vae", ext: "aimodel"),
              let tokURL = Self.resolveTokenizer(in: dir),
              let ehsURL = Self.resolveFile(in: dir, suffix: ".f32")
        else { throw Self.err("GLM bundle incomplete (need ar/dit .aimodelc, vae .aimodel, tokenizer, ehs.f32)") }

        (arFn, arD) = try await Self.loadModel(arURL, cpu: false)
        (ditFn, ditD) = try await Self.loadModel(ditURL, cpu: false)
        // Pick 512/1024 config from the DiT's latent side (hidden_states = [1,16,lat,lat], lat = size/8).
        if let dd = ditD {
            cfg = (descriptor(dd, "hidden_states", .input).shape[2] * 8 >= 1024) ? Self.cfg1024 : Self.cfg512
        }
        (vaeFn, vaeD) = try await Self.loadModel(vaeURL, cpu: true) // fp16 VAE overflows → fp32 CPU
        tokenizer = try await AutoTokenizer.from(modelFolder: tokURL)
        let data = try Data(contentsOf: ehsURL)
        ehs = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard ehs.count == 1472 else { throw Self.err("ehs \(ehs.count) != 1472") }
    }

    // MARK: Generation

    /// prompt → 512×512 image. `progress(step, total)` returns false to cancel (checked each DiT step).
    func generate(prompt: String, seed: UInt64, steps: Int = 20, guidance: Float = 1.5,
                  progress: @escaping @Sendable (Int, Int) -> Bool) async throws -> CGImage {
        let steps = max(4, min(steps, 100))   // sane clamp for the UI stepper
        guard let tokenizer, let arFn, let arD, let ditFn, let ditD, let vaeFn, let vaeD else {
            throw Self.err("GLM pipeline not loaded")
        }
        // 1) AR inputs (tokenize + size-specific suffix + host 3D positions).
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false) + cfg.suffix
        let L = ids.count
        let (decT, decH, decW) = decodePositions(startPos: L)
        let gs = cfg.grids.map { $0.0 * $0.1 }
        let maxNew = gs.reduce(0, +) + 1, offset = gs.dropFirst().reduce(0, +)
        let (th, tw) = cfg.grids[0]

        // 2) AR prefill + sampled decode → prior[1024].
        var rng = RNG(s: seed)
        var key = alloc(arD, "keyCache", stateShape(arD, "keyCache", dyn: kvSeq), .state)
        var val = alloc(arD, "valueCache", stateShape(arD, "valueCache", dyn: kvSeq), .state)
        fillF(&key, [Float](repeating: 0, count: key.shape.reduce(1, *)))
        fillF(&val, [Float](repeating: 0, count: val.shape.reduce(1, *)))

        func arStep(token: Int, t: Int, h: Int, w: Int, idx: Int) async throws -> [Float] {
            let (cs, sn) = arCosSin(t: t, h: h, w: w)
            var idA = alloc(arD, "input_ids", [1, 1], .input); fillI(&idA, [Int32(token)])
            var posA = alloc(arD, "position_ids", [1, idx + 1], .input); fillI(&posA, (0...idx).map { Int32($0) })
            var cosA = alloc(arD, "cos", [1, 1, rot], .input); fillF(&cosA, cs)
            var sinA = alloc(arD, "sin", [1, 1, rot], .input); fillF(&sinA, sn)
            var lg = alloc(arD, "logits", [1, 1, visionVocab], .output)
            var st = InferenceFunction.MutableViews(); st.insert(&key, for: "keyCache"); st.insert(&val, for: "valueCache")
            var ov = InferenceFunction.MutableViews(); ov.insert(&lg, for: "logits")
            _ = try await arFn.run(inputs: ["input_ids": idA, "position_ids": posA, "cos": cosA, "sin": sinA],
                                   states: consume st, outputViews: consume ov)
            return flattenAsFloat(lg)
        }

        var logits = [Float]()
        for i in 0..<L { logits = try await arStep(token: ids[i], t: i, h: i, w: i, idx: i) }  // prefill = ramp
        var gen = [sampleToken(logits, &rng)]
        for k in 0..<(maxNew - 1) {
            logits = try await arStep(token: gen[gen.count - 1], t: decT[k], h: decH[k], w: decW[k], idx: L + k)
            gen.append(sampleToken(logits, &rng))
        }
        // 2nd (large) block → th×tw → 2× nearest upsample → prior[1024].
        let block = Array(gen[offset..<(offset + th * tw)])
        var prior = [Int32](); prior.reserveCapacity(th * tw * 4)
        for r in 0..<(2 * th) { for c in 0..<(2 * tw) { prior.append(Int32(block[(r / 2) * tw + (c / 2)])) } }

        // 3) DiT 20-step CFG flow-match euler.
        let latS = side / 8, latN = 16 * latS * latS
        let (cos, sin) = ditRope(latH: latS, latW: latS)
        let tgt: [Float] = [Float(side), Float(side)], crop: [Float] = [0, 0]
        var lat = gaussian(latN, seed: seed)

        func dit(_ lat: [Float], ts: Float, scale: Float) async throws -> [Float] {
            var h = alloc(ditD, "hidden_states", [1, 16, latS, latS], .input); fillF(&h, lat)
            var e = alloc(ditD, "encoder_hidden_states", [1, 1, ehs.count], .input); fillF(&e, ehs)
            var pt = alloc(ditD, "prior_token_id", [1, prior.count], .input); fillI(&pt, prior)
            var ps = alloc(ditD, "prior_scale", [1, 1, 1], .input); fillF(&ps, [scale])
            var t = alloc(ditD, "timestep", [1], .input); fillF(&t, [ts])
            var tg = alloc(ditD, "target_size", [1, 2], .input); fillF(&tg, tgt)
            var cr = alloc(ditD, "crop_coords", [1, 2], .input); fillF(&cr, crop)
            var co = alloc(ditD, "cos", [latS / 2 * latS / 2, 128], .input); fillF(&co, cos)
            var si = alloc(ditD, "sin", [latS / 2 * latS / 2, 128], .input); fillF(&si, sin)
            var noise = alloc(ditD, "noise", [1, 16, latS, latS], .output)
            var ov = InferenceFunction.MutableViews(); ov.insert(&noise, for: "noise")
            _ = try await ditFn.run(
                inputs: ["hidden_states": h, "encoder_hidden_states": e, "prior_token_id": pt, "prior_scale": ps,
                         "timestep": t, "target_size": tg, "crop_coords": cr, "cos": co, "sin": si],
                states: InferenceFunction.MutableViews(), outputViews: consume ov)
            return flattenAsFloat(noise)
        }

        // Schedule: RAW integer timesteps condition the DiT (adaLN — must stay unshifted; feeding
        // the mu-shifted value skews the time conditioning every step → resolution-dependent color
        // drift). The latent euler steps use the mu-shifted sigmas. See `schedule(side:steps:)`.
        let (rawTs, sigmas) = Self.schedule(side: side, steps: steps)
        for i in 0..<steps {
            if !progress(i, steps) { throw CancellationError() }
            let ts = rawTs[i] - 1
            let nc = try await dit(lat, ts: ts, scale: 1)
            let nu = try await dit(lat, ts: ts, scale: 0)
            let dsig = sigmas[i + 1] - sigmas[i]
            for k in 0..<latN { lat[k] += dsig * (nu[k] + guidance * (nc[k] - nu[k])) }
        }

        // 4) VAE decode (fp32 CPU) → image[1,3,H,W].
        var z = alloc(vaeD, "z", [1, 16, latS, latS], .input); fillF(&z, lat)
        var img = alloc(vaeD, "image", [1, 3, side, side], .output)
        var ov = InferenceFunction.MutableViews(); ov.insert(&img, for: "image")
        _ = try await vaeFn.run(inputs: ["z": z], states: InferenceFunction.MutableViews(), outputViews: consume ov)
        guard let cg = Self.makeCGImage(rgb: flattenAsFloat(img), side: side) else { throw Self.err("VAE→CGImage failed") }
        return cg
    }

    // MARK: Host math (ported from GlmImageRunner)

    /// decode 3D-mrope positions: grids processed in reverse; t = block base, h += row, w += col, + end.
    private func decodePositions(startPos: Int) -> (t: [Int], h: [Int], w: [Int]) {
        var tl = [Int](), hl = [Int](), wl = [Int](); var dp = startPos
        for i in 1...cfg.grids.count {
            let (h, w) = cfg.grids[cfg.grids.count - i]
            for r in 0..<h { for c in 0..<w { tl.append(dp); hl.append(dp + r); wl.append(dp + c) } }
            dp += max(h, w)
        }
        tl.append(dp); hl.append(dp); wl.append(dp)
        return (tl, hl, wl)
    }

    /// AR mrope cos/sin for one 3D position → [64] each. channel k: axis = k<8 ? t : k<20 ? h : w.
    private func arCosSin(t: Int, h: Int, w: Int) -> (cos: [Float], sin: [Float]) {
        var cs = [Float](repeating: 0, count: rot), sn = cs
        for k in 0..<(rot / 2) {
            let axis = k < mrope[0] ? t : (k < mrope[0] + mrope[1] ? h : w)
            let ang = Double(axis) * inv[k]
            let c = Float(Foundation.cos(ang)), s = Float(Foundation.sin(ang))
            cs[k] = c; cs[k + rot / 2] = c; sn[k] = s; sn[k + rot / 2] = s
        }
        return (cs, sn)
    }

    /// DiT rope cos/sin per patch → [hw][128] flattened; j<32:r·inv, 32..63:c·inv, 64..95:r·inv, 96..127:c·inv.
    private func ditRope(latH: Int, latW: Int) -> (cos: [Float], sin: [Float]) {
        let h = latH / 2, w = latW / 2, dim = 128
        var cs = [Float](repeating: 0, count: h * w * dim), sn = cs
        for r in 0..<h {
            for c in 0..<w {
                let base = (r * w + c) * dim
                for j in 0..<dim {
                    let (axis, fi): (Int, Int) = j < 32 ? (r, j) : j < 64 ? (c, j - 32) : j < 96 ? (r, j - 64) : (c, j - 96)
                    let ang = Double(axis) * inv[fi]
                    cs[base + j] = Float(Foundation.cos(ang)); sn[base + j] = Float(Foundation.sin(ang))
                }
            }
        }
        return (cs, sn)
    }

    /// temp 0.9 / top-p 0.75 (GLM-Image generation_config). Greedy collapses visual tokens to flat
    /// regions; sampling restores detail. Inclusive top-p prefix + always the top token.
    private func sampleToken(_ logits: [Float], _ rng: inout RNG, temp: Double = 0.9, topP: Double = 0.75) -> Int {
        let n = logits.count
        var mx = -Double.infinity
        for v in logits { let d = Double(v) / temp; if d > mx { mx = d } }
        var p = [Double](repeating: 0, count: n); var sum = 0.0
        for i in 0..<n { let e = Foundation.exp(Double(logits[i]) / temp - mx); p[i] = e; sum += e }
        for i in 0..<n { p[i] /= sum }
        let order = Array(0..<n).sorted { p[$0] > p[$1] }
        var kept = [Int](); var csum = 0.0
        for (rank, idx) in order.enumerated() { csum += p[idx]; if rank == 0 || csum <= topP { kept.append(idx) } else { break } }
        var kp = kept.map { p[$0] }; let ks = kp.reduce(0, +)
        for i in 0..<kp.count { kp[i] /= ks }
        let r = rng.next(); var acc = 0.0
        for i in 0..<kp.count { acc += kp[i]; if r <= acc { return kept[i] } }
        return kept.last!
    }

    /// Box–Muller over SplitMix64 (host noise; deterministic per seed).
    private func gaussian(_ n: Int, seed: UInt64) -> [Float] {
        var rng = RNG(s: seed &+ 0xD1B54A32D192ED03)
        var out = [Float](repeating: 0, count: n); var i = 0
        while i < n {
            let u1 = max(rng.next(), 1e-12), u2 = rng.next()
            let r = (-2 * Foundation.log(u1)).squareRoot()
            out[i] = Float(r * Foundation.cos(2 * Double.pi * u2)); i += 1
            if i < n { out[i] = Float(r * Foundation.sin(2 * Double.pi * u2)); i += 1 }
        }
        return out
    }

    struct RNG {
        var s: UInt64
        mutating func next() -> Double {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            return Double(z >> 11) / Double(1 << 53)
        }
    }

    // MARK: NDArray helpers

    private enum IOKind { case input, state, output }
    private func descriptor(_ d: InferenceFunctionDescriptor, _ name: String, _ k: IOKind) -> NDArrayDescriptor {
        let io = switch k {
        case IOKind.input: d.inputDescriptor(of: name)
        case .state: d.stateDescriptor(of: name)
        case .output: d.outputDescriptor(of: name)
        }
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return nd
    }
    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ k: IOKind) -> NDArray {
        NDArray(descriptor: descriptor(d, name, k).resolvingDynamicDimensions(shape))
    }
    private func stateShape(_ d: InferenceFunctionDescriptor, _ name: String, dyn: Int) -> [Int] {
        descriptor(d, name, .state).shape.map { $0 < 0 ? dyn : $0 }
    }
    private func fillF(_ a: inout NDArray, _ v: [Float]) {
        switch a.scalarType {
        case .float16: fillNDArray(&a, as: Float16.self, with: v.map { Float16($0) })
        case .float32: fillNDArray(&a, as: Float.self, with: v)
        default: fatalError("fillF on \(a.scalarType)")
        }
    }
    private func fillI(_ a: inout NDArray, _ v: [Int32]) { fillNDArray(&a, as: Int32.self, with: v) }

    // MARK: Static helpers

    private static func loadModel(_ url: URL, cpu: Bool) async throws -> (InferenceFunction, InferenceFunctionDescriptor) {
        // AOT-compiled .aimodelc must load with its baked delegate (.default). Forcing a compute
        // unit (preferredComputeUnitKind:) re-specializes the graph via JIT, which wedges large
        // OS27 graphs → failedToSpecialize. (.default has preferredComputeUnitKind == nil; the
        // AR/DiT were AOT-compiled for GPU h16c, the VAE runs CPU-only in fp32.)
        let opts = cpu ? SpecializationOptions.cpuOnly : SpecializationOptions.default
        // Load the REAL path, not a symlink: AIModel(contentsOf:) does not follow a symlinked
        // bundle to its arch-specific AOT delegate subdir (main-h16c-delegates), so a symlinked
        // bundle fails with `failedToSpecialize`. resolvingSymlinksInPath fixes it.
        let m = try await AIModel(contentsOf: url.resolvingSymlinksInPath(), options: opts)
        guard let d = m.functionDescriptor(for: "main"), let f = try m.loadFunction(named: "main") else {
            throw Self.err("no 'main' function in \(url.lastPathComponent)")
        }
        return (f, d)
    }

    /// Image[3,H,W] (~[-1,1]) → clip(v/2+0.5) → RGBA8 → CGImage.
    private static func makeCGImage(rgb: [Float], side: Int) -> CGImage? {
        let plane = side * side
        var px = [UInt8](repeating: 255, count: plane * 4)
        for i in 0..<plane {
            for c in 0..<3 {
                let v = min(max(rgb[c * plane + i] / 2 + 0.5, 0), 1)
                px[i * 4 + c] = UInt8((v * 255).rounded())
            }
        }
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return px.withUnsafeMutableBytes { buf in
            CGContext(data: buf.baseAddress, width: side, height: side, bitsPerComponent: 8,
                      bytesPerRow: 4 * side, space: cs,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
    }

    private static func resolve(in dir: URL, contains: String, ext: String) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.pathExtension == ext && $0.lastPathComponent.lowercased().contains(contains) }
    }
    private static func resolveFile(in dir: URL, suffix: String) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.lastPathComponent.lowercased().hasSuffix(suffix) }
    }
    /// A subdir named tokenizer/processor holding tokenizer.json, else the folder itself if it has one.
    private static func resolveTokenizer(in dir: URL) -> URL? {
        let fm = FileManager.default
        for name in ["tokenizer", "processor"] {
            let sub = dir.appendingPathComponent(name, isDirectory: true)
            if fm.fileExists(atPath: sub.appendingPathComponent("tokenizer.json").path) { return sub }
        }
        return fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "GlmImagePipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
