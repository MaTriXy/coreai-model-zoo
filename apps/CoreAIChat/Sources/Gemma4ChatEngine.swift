// Gemma4ChatEngine — the RELEASE UI engine: user-selectable modes over the device-verified
// paths, one chat surface.
//
//   GPU    -> Gemma4MonolithBackend (metal-kernel host-cache monolith + head-argmax kernel, 8/8)
//   ANE    -> Gemma4ChunkBackend    (6-chunk host-cache + on-ANE argmax head, 8/8)
//   Gemma⚡ -> PipelinedBackend(.gemmaTbl) (Gemma 4 E2B int4lin, PLE table as static input)
//   Qwen   -> PipelinedBackend(.qwen) (Qwen3.5-0.8B int8hu, coreai-pipelined engine)
//   LFM    -> PipelinedBackend(.lfm2) (LFM2.5-1.2B int8lin, coreai-pipelined engine)
//
// Mode switch frees the current backend's models before loading the other (the sets together
// approach the jetsam ceiling). Each generate() resets the host-owned KV and prefills the current
// user message only (single-turn semantics, same as the verified harnesses).
//
// Headless hooks (devicectl process launch --console --environment-variables):
//   GEMMA_ENGINE = gpu (default) | ane | gemmatbl | qwen | lfm2 ...   initial mode
//   GEMMA_VERIFY = 1                        drive the HF-oracle ids through the SELECTED release
//                                           engine, print match n/8 (the release-path 8/8 check)
//   GEMMA_PROMPT / GEMMA_N                  generate + print output, tok/s, per-token profile

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers

// Common surface of the two release backends. Both are @MainActor classes that own their models
// and host-side KV; `step` runs one token at absolute position `pos` and returns the argmax token
// (or -1 when `needToken` is false and the head dispatch was skipped — prefill positions whose
// logits nobody reads).
@MainActor
protocol Gemma4Backend: AnyObject {
    var modeLabel: String { get }       // "GPU" | "ANE"
    var ctx: Int { get }                // fixed cache capacity (bucket) — prompt+gen must fit
    func load() async throws
    func reset()                        // zero KV + profile counters (new conversation turn)
    func step(_ tok: Int32, _ pos: Int, needToken: Bool) async throws -> Int
    func profileSummary() -> String     // per-token ms breakdown of the last generation
}

/// The MODEL level of the picker. GPU/ANE is a compute-unit toggle one level
/// below this (gemma only — the pipelined models are GPU-only by design).
enum ChatModel: String, CaseIterable, Identifiable {
    case gemma = "Gemma 4 E2B", qwen = "Qwen3.5 0.8B", qwen2b = "Qwen3.5 2B",
         lfm2 = "LFM2.5 1.2B", granite = "Granite 1B", minicpm5 = "MiniCPM5 1B",
         qwen3vl = "Qwen3-VL 2B", qwen3vl4b = "Qwen3-VL 4B", gemma4vl = "Gemma 4 VL"
    var id: String { rawValue }
}

// Engine-selection enum (model × compute unit flattened): the storage type the
// whole engine layer is keyed on, and the headless GEMMA_ENGINE vocabulary
// (gpu / ane / gemmatbl / qwen / qwen2b / lfm2 / granite). The UI splits it
// into ChatModel + a gemma-only GPU/ANE/⚡ segment.
enum GemmaMode: String, CaseIterable, Identifiable {
    case gpu = "GPU", ane = "ANE", gemmaTbl = "Gemma⚡", qwen = "Qwen",
         qwen2b = "Qwen2B", lfm2 = "LFM", granite = "Granite", minicpm5 = "MiniCPM5",
         qwen3vl = "Qwen3VL", qwen3vl4b = "Qwen3VL4B", gemma4vl = "Gemma4VL"
    var id: String { rawValue }
    /// The model family this engine mode belongs to (the picker's top level).
    var chatModel: ChatModel {
        switch self {
        case .gpu, .ane, .gemmaTbl: .gemma
        case .qwen: .qwen
        case .qwen2b: .qwen2b
        case .lfm2: .lfm2
        case .granite: .granite
        case .minicpm5: .minicpm5
        case .qwen3vl: .qwen3vl
        case .qwen3vl4b: .qwen3vl4b
        case .gemma4vl: .gemma4vl
        }
    }
    /// User-facing label for the download panel (model, plus unit where it matters).
    var downloadLabel: String {
        switch self {
        case .gpu: "Gemma 4 E2B · GPU"
        case .ane: "Gemma 4 E2B · ANE"
        case .gemmaTbl: "Gemma 4 E2B · ⚡"
        case .qwen: "Qwen3.5 0.8B"
        case .qwen2b: "Qwen3.5 2B"
        case .lfm2: "LFM2.5 1.2B"
        case .granite: "Granite 4.0-H 1B"
        case .minicpm5: "MiniCPM5 1B"
        case .qwen3vl: "Qwen3-VL 2B (vision)"
        case .qwen3vl4b: "Qwen3-VL 4B (vision)"
        case .gemma4vl: "Gemma 4 E2B VL (vision)"
        }
    }
    /// Non-nil for the modes that ride the coreai-pipelined engine.
    var pipelinedSpec: PipelinedBackend.Spec? {
        switch self {
        case .gemmaTbl: PipelinedBackend.gemmaTbl
        case .qwen: PipelinedBackend.qwen
        case .qwen2b: PipelinedBackend.qwen2b
        case .lfm2: PipelinedBackend.lfm2
        case .granite: PipelinedBackend.granite
        case .minicpm5: PipelinedBackend.minicpm5
        case .gpu, .ane, .qwen3vl, .qwen3vl4b, .gemma4vl: nil  // the VLMs drive their own backends
        }
    }
    /// Non-nil for the Qwen3-VL modes (own backend with a fixed-grid vision tower).
    var qwen3vlSpec: Qwen3VLBackend.Spec? {
        switch self {
        case .qwen3vl: Qwen3VLBackend.qwen3vl2b
        case .qwen3vl4b: Qwen3VLBackend.qwen3vl4b
        default: nil
        }
    }
    /// VLM modes (photo picker + image attach surface).
    var isVL: Bool { self == .qwen3vl || self == .qwen3vl4b || self == .gemma4vl }
}

@MainActor
final class Gemma4ChatEngine: ObservableObject {
    @Published var mode: GemmaMode
    @Published var status = "starting…"
    @Published var ready = false
    @Published var loading = false
    @Published var busy = false
    @Published var output = ""
    @Published var stats = ""

    private var backend: Gemma4Backend?
    private var pipelined: PipelinedBackend?  // .qwen / .lfm2 (pipelined engine, own loop)
    private var vl: Qwen3VLBackend?           // .qwen3vl (pipelined engine + vision tower)
    private var gvl: Gemma4VLBackend?         // .gemma4vl (provider mode + vision tower)
    @Published var vlImageAttached = false
    private var loadedMode: GemmaMode?  // mode the current backend was loaded for
    private var tokenizer: Tokenizer!
    private let EOT = 106               // gemma <end_of_turn>
    private var eosId = 106

    init() {
        switch ProcessInfo.processInfo.environment["GEMMA_ENGINE"] {
        case "ane": mode = .ane
        case "gemmatbl": mode = .gemmaTbl
        case "qwen": mode = .qwen
        case "qwen2b": mode = .qwen2b
        case "lfm2", "lfm": mode = .lfm2
        case "granite": mode = .granite
        case "minicpm5", "mc5": mode = .minicpm5
        case "qwen3vl", "vl": mode = .qwen3vl
        case "qwen3vl4b", "vl4b": mode = .qwen3vl4b
        case "gemma4vl", "gvl": mode = .gemma4vl
        default: mode = .gpu
        }
    }

    private func docs() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }

    // MARK: model delivery

    // Where the published artifact set for each mode lives (GEMMA_REPO / the UI field override it).
    static func defaultRepo(for mode: GemmaMode) -> String {
        switch mode {
        case .qwen: "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI"
        case .qwen2b: "https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI"
        case .lfm2: "https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI"
        case .granite: "https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI"
        case .minicpm5: "https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI"
        case .qwen3vl: "https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI"
        case .qwen3vl4b: "https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI"
        case .gpu, .ane, .gemmaTbl, .gemma4vl: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI"
        }
    }

    // Download targets (repo subpath -> name under Documents/models) the CURRENT mode still
    // needs. Empty == everything is on disk. The gemma tokenizer ships in the app bundle (qwen's
    // is inside its bundle), and the env overrides (GEMMA_MONO_MODEL/HEAD, GEMMA_QUANT) are dev
    // paths not covered here.
    // Every (repo subpath -> name under Documents/models) artifact a mode needs, present or not.
    private func modelPaths(for mode: GemmaMode) -> [(remote: String, local: String)] {
        switch mode {
        case .qwen, .qwen2b, .lfm2, .granite, .minicpm5:
            let spec = mode.pipelinedSpec!
            return [(spec.hfRemotePath, spec.bundleName)]
        case .qwen3vl, .qwen3vl4b:
            let spec = mode.qwen3vlSpec!
            return [(spec.hfDecoderPath, spec.decoderBundle),
                    (spec.hfVisionPath, spec.visionDir)]
        case .gemma4vl:
            // decoder (provider aotc) + vision tower + the QAT PLE table set
            return [(Gemma4VLBackend.hfDecoderPath, Gemma4VLBackend.decoderBundle),
                    (Gemma4VLBackend.hfVisionPath, Gemma4VLBackend.visionDir),
                    (Gemma4VLBackend.hfTablesPath, Gemma4VLBackend.tablesDir)]
        case .gemmaTbl:
            // AOT engine bundle + the PLE table set the GPU/ANE modes already share.
            let spec = mode.pipelinedSpec!
            return [(spec.hfRemotePath, spec.bundleName),
                    ("ios-frontend/gemma4_gather_raw", "gemma4_gather_raw")]
        case .gpu:
            return [("ios-frontend/gemma4_gather_raw", "gemma4_gather_raw"),
                    ("ios-gpu/gemma4_e2b_metal_int4km_L35.aimodel", "gemma4_e2b_metal_int4km_L35.aimodel"),
                    ("ios-gpu/gemma4_e2b_head_argmax_int4km.aimodel", "gemma4_e2b_head_argmax_int4km.aimodel")]
        case .ane:
            var paths: [(remote: String, local: String)] =
                [("ios-frontend/gemma4_gather_raw", "gemma4_gather_raw")]
            paths += (1...6).map { ("ios-ane/gemma4_e2b_hostcache_chunk\($0)_int8.aimodel",
                                    "gemma4_e2b_hostcache_chunk\($0)_int8.aimodel") }
            paths += [("ios-ane/gemma4_e2b_head_argmax_int8.aimodel", "gemma4_e2b_head_argmax_int8.aimodel")]
            return paths
        }
    }

    // Download targets the CURRENT mode still needs. Empty == everything is on disk.
    func missingDownloads() -> [ModelDownloader.Item] {
        let models = docs().appendingPathComponent("models")
        return modelPaths(for: mode)
            .filter { !FileManager.default.fileExists(atPath: models.appendingPathComponent($0.local).path) }
            .map { ModelDownloader.Item(remote: $0.remote, local: $0.local) }
    }

    // Headless model-download driver (GEMMA_DL_TEST=1) — installs the current mode's missing files
    // without the UI, mirroring the Download button (device provisioning / CI).
    func downloadSelfTest() async {
        let items = missingDownloads()
        print("[dl-test] mode=\(mode) missing=\(items.count)")
        guard !items.isEmpty else { print("[dl-test] already installed"); return }
        let dl = ModelDownloader()
        let probe = Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 6_000_000_000)
                print("[dl-test] \(Int(dl.fraction * 100))% \(dl.detail)")
            }
        }
        await dl.fetch(repo: Self.defaultRepo(for: mode), items: items, into: docs().appendingPathComponent("models"))
        probe.cancel()
        print("[dl-test] done: \(dl.phase)")
    }

    // True when every file the current mode needs is on disk (its chat-ready set is installed).
    var installedForCurrentMode: Bool { missingDownloads().isEmpty }

    private func isInstalled(_ m: GemmaMode) -> Bool {
        let models = docs().appendingPathComponent("models")
        return modelPaths(for: m).allSatisfy {
            FileManager.default.fileExists(atPath: models.appendingPathComponent($0.local).path)
        }
    }

    // Files of the current mode safe to delete: present on disk and not still shared by another
    // mode that is itself fully installed — so the shared `gemma4_gather_raw` PLE table is kept
    // while ANE/⚡ stay installed and only freed once they're gone too.
    private func deletableLocalDirs() -> [String] {
        let models = docs().appendingPathComponent("models")
        let protected = Set(
            GemmaMode.allCases
                .filter { $0 != mode && isInstalled($0) }
                .flatMap { modelPaths(for: $0).map(\.local) }
        )
        return modelPaths(for: mode)
            .map(\.local)
            .filter { local in
                !protected.contains(local)
                    && FileManager.default.fileExists(atPath: models.appendingPathComponent(local).path)
            }
    }

    // Remove the current mode's downloaded files from this device (always re-downloadable). Frees
    // the loaded backend first so nothing is memory-mapped, and keeps files another installed mode
    // still shares. The mode then drops back to its download panel.
    func deleteCurrentModel() async {
        backend = nil
        pipelined?.unload(); pipelined = nil
        vl?.unload(); vl = nil
        gvl?.unload(); gvl = nil
        vlImageAttached = false
        loadedMode = nil
        ready = false
        let models = docs().appendingPathComponent("models")
        for local in deletableLocalDirs() {
            try? FileManager.default.removeItem(at: models.appendingPathComponent(local))
        }
        output = ""; stats = ""
        status = "\(mode.downloadLabel) removed"
    }

    // Load (or re-load after a mode switch) the backend for the current mode.
    // A concurrent call (UI onChange + headless autoTest both react to a mode change) waits for the
    // in-flight load; after the wait it re-checks the target mode, so a rapid GPU→ANE→GPU flip still
    // converges on the last selected mode.
    func load() async {
        while loading { try? await Task.sleep(nanoseconds: 100_000_000) }
        if loadedMode == mode, ready { return }
        loading = true; ready = false
        // Free the previous mode's models BEFORE loading the next set (jetsam headroom).
        backend = nil; pipelined?.unload(); pipelined = nil
        vl?.unload(); vl = nil; gvl?.unload(); gvl = nil
        vlImageAttached = false; loadedMode = nil
        let target = mode
        // Files for this mode aren't on disk yet — don't attempt a load that can only fail with a
        // confusing "load error"; leave the download panel up with a clear status instead. (Called
        // again after the download completes, when the files are present.)
        guard installedForCurrentMode else {
            status = "\(target.downloadLabel) — tap Download to install"
            loading = false
            return
        }
        status = "loading \(target.rawValue) engine…"
        do {
            let tLoad = Date()
            if let vlSpec = target.qwen3vlSpec {
                let b = Qwen3VLBackend(spec: vlSpec)
                try await b.load()
                vl = b
                loadedMode = target
                ready = true
                status = "\(b.spec.label) ready · ctx \(b.ctx)"
            } else if target == .gemma4vl {
                let b = Gemma4VLBackend()
                try await b.load()
                gvl = b
                loadedMode = target
                ready = true
                status = "\(Gemma4VLBackend.label) ready · ctx \(b.ctx)"
            } else if let spec = target.pipelinedSpec {
                let pb = PipelinedBackend(spec: spec)
                try await pb.load()
                pipelined = pb
                loadedMode = target
                ready = true
                status = "\(spec.label) ready · ctx \(pb.ctx)"
            } else {
                let be: Gemma4Backend = (target == .gpu) ? Gemma4MonolithBackend() : Gemma4ChunkBackend()
                try await be.load()
                if tokenizer == nil {
                    // Bundled tokenizer first; a sideloaded Documents/tokenizer (devicectl) wins if present.
                    let sideloaded = docs().appendingPathComponent("tokenizer")
                    let folder = FileManager.default.fileExists(atPath: sideloaded.path)
                        ? sideloaded
                        : (Bundle.main.url(forResource: "tokenizer", withExtension: nil) ?? sideloaded)
                    tokenizer = try await AutoTokenizer.from(modelFolder: folder, strict: false)
                    eosId = tokenizer.eosTokenId ?? EOT
                }
                backend = be
                loadedMode = target
                ready = true
                status = "\(target.rawValue) ready · ctx \(be.ctx)"
            }
            print(String(format: "[chat] %@ engine load %.1fs", target.rawValue, -tLoad.timeIntervalSinceNow))
        } catch {
            status = "load error: \(error.localizedDescription)"
            print("[chat] \(target.rawValue) load error: \(error)")
        }
        loading = false
        // The selection moved while we were loading (rapid flip) — converge on the latest mode.
        if mode != target { await load() }
    }

    // q1 prefill (head only at the last prompt position) + greedy decode, streamed to `output`.
    func generate(_ prompt: String, maxNew: Int? = nil) async {
        guard ready, !busy else { return }
        if mode.isVL {
            await generateVL(prompt, maxNew: maxNew)
            return
        }
        if mode.pipelinedSpec != nil {
            await generatePipelined(prompt, maxNew: maxNew)
            return
        }
        guard let be = backend else { return }
        busy = true; output = ""; stats = ""
        defer { busy = false }
        do {
            let ids = (try tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]])).map { Int32($0) }
            guard ids.count < be.ctx else {
                output = "prompt (\(ids.count) tok) does not fit ctx \(be.ctx)"; return
            }
            // Same budget rule as the CoreML-LLM chat app: spend the remaining ctx,
            // soft-capped to avoid long hangs.
            let budget = maxNew ?? min(be.ctx - ids.count - 1, 1024)
            be.reset()
            let tPre = Date(); var last = 0
            for (pos, t) in ids.enumerated() {
                last = try await be.step(t, pos, needToken: pos == ids.count - 1)
            }
            let preSec = -tPre.timeIntervalSinceNow

            var gen: [Int] = []
            var pos = ids.count
            let tDec = Date()
            // Throttle the live re-decode + view update to ~25 fps and exclude its time
            // from the decode timer (re-decoding the whole token list every token is O(n²)).
            var uiSec = 0.0
            var lastEmit = tDec
            if last != eosId && last != EOT {
                gen.append(last)
                output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)
            }
            // Positions 0..ctx-1 are valid cache columns — stop before overrunning the bucket.
            while gen.count < budget, last != eosId, last != EOT, pos < be.ctx {
                last = try await be.step(Int32(last), pos, needToken: true)
                pos += 1
                if last == eosId || last == EOT { break }
                gen.append(last)
                if Date().timeIntervalSince(lastEmit) >= 0.04 {
                    let u = Date()
                    output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)  // live stream
                    uiSec += -u.timeIntervalSinceNow
                    lastEmit = Date()
                }
            }
            let uF = Date()
            output = tokenizer.decode(tokens: gen, skipSpecialTokens: true)  // final flush
            uiSec += -uF.timeIntervalSinceNow
            let decSec = -tDec.timeIntervalSinceNow - uiSec
            let full = pos >= be.ctx ? " · ctx full" : ""
            stats = String(format: "%@ · prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s%@ | %@",
                           be.modeLabel,
                           ids.count, Double(ids.count) / max(preSec, 1e-6),
                           gen.count, Double(gen.count) / max(decSec, 1e-6),
                           full, be.profileSummary())
        } catch {
            output = "generation error: \(error)"
        }
    }

    // Pipelined path (qwen / lfm2): the engine streams its own loop (async encode +
    // on-GPU argmax), so generation is a stream consumption, not a host step() loop.
    private func generatePipelined(_ prompt: String, maxNew: Int?) async {
        guard let pb = pipelined else { return }
        busy = true; output = ""; stats = ""
        defer { busy = false }
        do {
            let st = try await pb.generate(prompt, maxNew: maxNew ?? 1024) { [weak self] text in
                self?.output = text  // live stream
            }
            stats = st.summary
        } catch {
            output = "generation error: \(error.localizedDescription)"
        }
    }

    private func generateVL(_ prompt: String, maxNew: Int?) async {
        busy = true; output = ""; stats = ""
        defer { busy = false }
        do {
            let st: PipelinedBackend.GenStats
            if let vl {
                st = try await vl.generate(prompt, maxNew: maxNew ?? 1024) { [weak self] text in
                    self?.output = text
                }
            } else if let gvl {
                st = try await gvl.generate(prompt, maxNew: maxNew ?? 1024) { [weak self] text in
                    self?.output = text
                }
            } else { return }
            stats = st.summary
        } catch {
            output = "generation error: \(error.localizedDescription)"
        }
    }

    /// Attach a picked photo to the VL conversation (runs the vision tower once).
    func attachVLImage(_ cgImage: CGImage) async {
        guard ready, mode.isVL else { return }
        busy = true
        defer { busy = false }
        do {
            if let vl {
                try await vl.attach(cgImage: cgImage)
                status = "\(vl.spec.label) ready · image attached"
            } else if let gvl {
                try await gvl.attach(cgImage: cgImage)
                status = "\(Gemma4VLBackend.label) ready · image attached"
            } else { return }
            vlImageAttached = true
        } catch {
            status = "image attach failed: \(error.localizedDescription)"
        }
    }

    func detachVLImage() {
        vl?.detachImage()
        gvl?.detachImage()
        vlImageAttached = false
        if let vl { status = "\(vl.spec.label) ready · ctx \(vl.ctx)" }
        if let gvl { status = "\(Gemma4VLBackend.label) ready · ctx \(gvl.ctx)" }
    }

    // HF-greedy oracle ("Count from one to ten") through the CURRENT release backend -> match n/8.
    private func verifyOracle() async {
        guard let be = backend, ready else { print("[chat] verify skipped (not ready)"); return }
        let prompt: [Int32] = [2, 105, 2364, 107, 4377, 699, 886, 531, 3595, 236764, 15914, 684, 162760, 236761, 106, 107, 105, 4368, 107]
        let ref = [4906, 236764, 1156, 236764, 1806, 236764, 2390, 236764]
        do {
            be.reset()
            var last = 0
            for (pos, t) in prompt.enumerated() {
                last = try await be.step(t, pos, needToken: pos == prompt.count - 1)
            }
            var gen = [last]
            for i in 0..<(ref.count - 1) {
                last = try await be.step(Int32(last), prompt.count + i, needToken: true)
                gen.append(last)
            }
            let n = zip(gen, ref).filter { $0 == $1 }.count
            print("[chat] mode=\(mode.rawValue) decode = \(gen)")
            print("[chat] mode=\(mode.rawValue) ref    = \(ref)")
            print("[chat] mode=\(mode.rawValue) verify match = \(n)/\(ref.count) -> \(n == ref.count ? "PASS ✅ EXACT" : "DEGRADED ⚠️")")
        } catch { print("[chat] mode=\(mode.rawValue) verify error: \(error)") }
    }

    // Headless self-test driver (see header). VERIFY runs the HF-greedy oracle ids through the
    // selected RELEASE engine — the device 8/8 evidence for the shipping path, not just the harness.
    // GEMMA_VERIFY_BOTH=1 additionally flips the mode IN-PROCESS (the real picker scenario: free the
    // current models, load the other set) and verifies again — proves the switch doesn't jetsam.
    func autoTestIfRequested() async {
        let env = ProcessInfo.processInfo.environment
        if env["GEMMA_VERIFY"] == "1" || env["GEMMA_VERIFY_BOTH"] == "1" {
            await verifyOracle()
            if env["GEMMA_VERIFY_BOTH"] == "1" {
                let other: GemmaMode = mode == .gpu ? .ane : .gpu
                print("[chat] switching mode \(mode.rawValue) -> \(other.rawValue) (in-process, frees current models)")
                mode = other
                await load()
                print("[chat] after switch: \(status)")
                await verifyOracle()
            }
        }
        if let p = env["GEMMA_PROMPT"], ready {
            print("[chat] mode=\(mode.rawValue) autotest prompt: \(p)")
            await generate(p, maxNew: Int(env["GEMMA_N"] ?? "24") ?? 24)
            print("[chat] OUT >>> \(output)")
            print("[chat] \(stats)")
        }
    }
}
