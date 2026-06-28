// MiniCPMVIEngine.swift — the single on-device VLM engine shared by the in-app "Try it" UI and
// the system Visual Intelligence query. THIS is "your model behind Visual Intelligence": the
// converted MiniCPM-V-4.6 bundle (vision SigLIP + qwen3.5-hybrid decode) run through the Core AI
// pipelined engine (`MiniCPMVLBackend`), fully on-device. Apple's VI "ask" sends the photo to
// ChatGPT in the cloud; this answers the same surface with a model on the A19 GPU, offline.
//
// The bundles are DOWNLOADED once from Hugging Face (mlboydaisuke/MiniCPM-V-4.6-CoreAI) into
// <Documents>/models by ModelDownloader, driven by the foreground app. The Visual Intelligence
// query runs in a background launch that has no bandwidth to download — so it only loads what the
// foreground already cached (exactly why the app prompts you to download + warm it first).
//
// The model state lives in `MiniCPMVLBackend`, which is `@MainActor` (it owns a Metal buffer and
// the Core AI engine). To stay usable from a *nonisolated* `IntentValueQuery.values(for:)` this
// engine is plain `@unchecked Sendable`: it isolates every backend touch to the main actor and
// exposes a pixels-based caption entry so the query never has to send a (non-Sendable) `CGImage`
// across the actor boundary — it preprocesses to a Sendable `[Float16]` first.

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox

final class MiniCPMVIEngine: @unchecked Sendable {
    /// Shared instance — registered as an App Intents dependency and used by the UI. In a
    /// background Visual Intelligence launch this is simply this process's instance.
    static let shared = MiniCPMVIEngine()

    /// One short sentence is the right shape for a Visual Intelligence row.
    static let defaultPrompt = "What is in this image? Answer in one short sentence."

    // MARK: - Model delivery (download spec, shared by the UI and the background query)

    static let repo = "https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI"
    /// The two bundles to fetch (decoder + tokenizer, and the fixed-grid SigLIP vision tower).
    /// `local` names match `MiniCPMVLBackend.decoderBundle` / `.visionDir`.
    static let items = [
        ModelDownloader.Item(
            remote: "gpu-pipelined/minicpmv46_vlm_decode_int8lin",
            local: MiniCPMVLBackend.decoderBundle),
        ModelDownloader.Item(
            remote: "gpu-pipelined/minicpmv46_vision", local: MiniCPMVLBackend.visionDir),
    ]

    /// <Documents>/models — same per-app container in the foreground and the background VI launch.
    static var modelsDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("models")
    }

    /// Both bundles are placed atomically by ModelDownloader, so presence == complete + ready.
    static var bundlesPresent: Bool {
        let fm = FileManager.default
        return items.allSatisfy {
            fm.fileExists(atPath: modelsDir.appendingPathComponent($0.local).path)
        }
    }

    /// Persisted A/B toggle the background VI process reads — flip it on device with no rebuild.
    /// OFF (default) = VI returns a light "tap to ask" teaser and the answer is produced in the
    /// foreground (full memory budget). ON = run MiniCPM-V directly inside the background VI
    /// launch so the answer surfaces with the app closed.
    static let runInVIKey = "runInVisualIntelligence"
    static var runInVisualIntelligence: Bool {
        get { UserDefaults.standard.bool(forKey: runInVIKey) }
        set { UserDefaults.standard.set(newValue, forKey: runInVIKey) }
    }

    // Backend + load memoization, all isolated to the main actor (the backend is @MainActor).
    @MainActor private var backend: MiniCPMVLBackend?
    @MainActor private var loadTask: Task<MiniCPMVLBackend, Error>?
    /// One captioning at a time: `attach` overwrites a single static image buffer, so two
    /// concurrent queries would corrupt each other. A second caller fails fast (degrades to teaser).
    @MainActor private var inFlight = false

    // MARK: - Load (memoized; requires the bundles already downloaded)

    @MainActor
    func ready(onStatus: ((String) -> Void)? = nil) async throws -> MiniCPMVLBackend {
        if let backend, backend.loaded { return backend }
        if let loadTask { return try await loadTask.value }
        guard Self.bundlesPresent else { throw Self.err("model not downloaded yet") }
        onStatus?("loading MiniCPM-V 4.6…")
        let dir = Self.modelsDir
        let task = Task { @MainActor () -> MiniCPMVLBackend in
            let b = backend ?? MiniCPMVLBackend()
            backend = b
            if !b.loaded { try await b.load(modelsDir: dir) }
            return b
        }
        loadTask = task
        do {
            return try await task.value
        } catch {
            loadTask = nil  // let a later attempt retry a failed load
            throw error
        }
    }

    @MainActor var isLoaded: Bool { backend?.loaded ?? false }

    // MARK: - Captioning

    /// Foreground (in-app) path: the caller already holds a `CGImage` on the main actor.
    /// `onUpdate` streams the partial answer for a live UI.
    @MainActor
    func caption(
        cgImage: CGImage, prompt: String = defaultPrompt, maxNew: Int = 96,
        onStatus: ((String) -> Void)? = nil, onUpdate: ((String) -> Void)? = nil
    ) async throws -> String {
        guard !inFlight else { throw Self.err("busy") }
        inFlight = true
        defer { inFlight = false }
        let b = try await ready(onStatus: onStatus)
        onStatus?("encoding image…")
        try await b.attach(cgImage: cgImage)
        onStatus?("answering on-device…")
        var out = ""
        try await b.generate(prompt, maxNew: maxNew) { out = $0; onUpdate?($0) }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Visual Intelligence path: takes Sendable preprocessed pixels (see `pixels(from:)`), so a
    /// nonisolated query can call it without sending a `CGImage` to the main actor.
    @MainActor
    func caption(pixels: [Float16], prompt: String = defaultPrompt, maxNew: Int = 64) async throws
        -> String
    {
        guard !inFlight else { throw Self.err("busy") }
        inFlight = true
        defer { inFlight = false }
        let b = try await ready()
        try await b.attach(pixels: pixels)
        var out = ""
        try await b.generate(prompt, maxNew: maxNew) { out = $0 }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Pixel buffer / image helpers (nonisolated — run in the query, off the main actor)

    /// Converts the Visual Intelligence `CVReadOnlyPixelBuffer` to a `CGImage` (format-agnostic).
    static func cgImage(from pixelBuffer: CVReadOnlyPixelBuffer) -> CGImage? {
        pixelBuffer.withUnsafeBuffer { buffer -> CGImage? in
            var out: CGImage?
            _ = VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &out)
            return out
        }
    }

    /// CGImage -> the model's Sendable input pixels (448×448 CHW f16), so the VI query can hand
    /// the backend only Sendable values. Reuses the backend's exact preprocessing.
    static func pixels(from cgImage: CGImage) -> [Float16] {
        MiniCPMVLBackend.preprocess(cgImage: cgImage, side: 448)
    }

    /// Small JPEG thumbnail (Sendable) for an entity's `DisplayRepresentation` image.
    static func jpeg(from image: CGImage, maxSide: Int = 256) -> Data? {
        let w = image.width
        let h = image.height
        let longest = max(w, h)
        let scale = longest > maxSide ? Double(maxSide) / Double(longest) : 1
        let nw = max(1, Int(Double(w) * scale))
        let nh = max(1, Int(Double(h) * scale))
        guard
            let ctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        guard let small = ctx.makeImage() else { return nil }
        let data = NSMutableData()
        guard
            let dest = CGImageDestinationCreateWithData(
                data as CFMutableData, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(
            dest, small, [kCGImageDestinationLossyCompressionQuality: 0.7] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    static func err(_ m: String) -> Error {
        NSError(domain: "MiniCPMVIEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
