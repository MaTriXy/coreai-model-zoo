// BenchRunner — the community field-data benchmark core (Bench tab + BENCH_AUTO headless).
//
// Trust lives in the harness, not the submitter: the app measures, the app builds the
// result blob, and no hand-entered number exists anywhere in the pipeline. The protocol
// is FIXED and baked into the blob (schema_version + protocol block): a splitmix64(seed 0)
// random prompt of 128 tokens, 256 greedy decode tokens, S=1 prefill
// (COREAI_CHUNK_THRESHOLD=1), 1 cold + 3 warm runs on a freshly created engine.
// Sloppy environments (background apps, hot device, Low Power Mode) are not fraud — they
// surface as outliers and are filtered downstream via the environment metadata
// (zoo scripts/aggregate_bench.py). Consecutive runs are allowed to throttle: we record
// the thermal state per run instead of hiding it behind cooldown waits.
//
// Measurement mirrors ondevice/PipelinedBench (the STATS prefill/decode standard): raw
// token ids through EngineFactory, no tokenizer, no text decode, no UI work in the timed
// loop. Bench models are the cataloged lightweight set only (must finish on <6 GB
// devices), pinned below — the repo/bundle is not user-editable.

import CoreAILanguageModels
import CoreAIShared
import Foundation
import UIKit

// MARK: - Bench catalog (pinned; the trust anchor for what was measured)

struct BenchModel: Identifiable, Hashable {
    let id: String              // stable row key in the community matrix
    let label: String           // UI name
    let repo: String            // HF repo slug (owner/name), pinned
    let remotePath: String      // subpath inside the repo (download source)
    let bundleName: String      // dir under Documents/models/
    let approxDownloadGB: Double

    static let catalog: [BenchModel] = [
        BenchModel(id: "qwen3.5-0.8b", label: "Qwen3.5 0.8B",
                   repo: "mlboydaisuke/qwen3.5-0.8B-CoreAI",
                   remotePath: PipelinedBackend.qwen.hfRemotePath,
                   bundleName: PipelinedBackend.qwen.bundleName,
                   approxDownloadGB: 1.3),
        BenchModel(id: "lfm2.5-1.2b", label: "LFM2.5 1.2B",
                   repo: "mlboydaisuke/LFM2.5-1.2B-CoreAI",
                   remotePath: PipelinedBackend.lfm2.hfRemotePath,
                   bundleName: PipelinedBackend.lfm2.bundleName,
                   approxDownloadGB: 1.5),
        BenchModel(id: "granite-4.0-h-1b", label: "Granite 4.0-H 1B",
                   repo: "mlboydaisuke/granite-4.0-h-CoreAI",
                   remotePath: PipelinedBackend.granite.hfRemotePath,
                   bundleName: PipelinedBackend.granite.bundleName,
                   approxDownloadGB: 1.2),
    ]

    var bundleDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models").appendingPathComponent(bundleName)
    }
    var isInstalled: Bool { FileManager.default.fileExists(atPath: bundleDir.path) }
}

// MARK: - Result blob schema (schema_version 1)

struct BenchBlob: Codable {
    var schema_version: Int
    var kind: String            // "coreai-community-bench"

    struct Device: Codable {
        var model_identifier: String   // utsname.machine, e.g. "iPhone18,1"
        var os: String                 // "iOS 27.1"
        var os_build: String           // kern.osversion, e.g. "27B84"
        var memory_gb: Double          // physical RAM (rounded)
    }
    struct AppInfo: Codable {
        var version: String
        var build: String
    }
    struct Model: Codable {
        var id: String
        var hf_repo: String
        var hf_revision: String        // main sha at bench time, or "unknown" (offline)
        var bundle: String
        var bundle_kind: String        // "aimodelc" (AOT) | "aimodel" (JIT)
    }
    struct BenchProtocol: Codable {
        var name: String               // "pb-random-v1"
        var prompt_tokens: Int
        var max_tokens: Int
        var prompt_seed: Int
        var temperature: Double
        var chunk_threshold: Int       // actual env value at run time (must be 1)
        var cold_runs: Int
        var warm_runs: Int
    }
    struct Environment: Codable {
        var thermal_state_before: String
        var thermal_state_after: String
        var low_power_mode: Bool
        var battery_level: Double      // 0..1, -1 = unknown
        var battery_state: String      // unplugged | charging | full | unknown
        var available_memory_mb: Int   // os_proc_available_memory at start
        var free_disk_gb: Double
    }
    struct Run: Codable {
        var kind: String               // "cold" | "warm"
        var prefill_tok_s: Double
        var decode_tok_s: Double
        var prefill_s: Double
        var decode_s: Double
        var gen_tokens: Int
        var thermal_state_end: String  // per-run throttle visibility
    }
    struct Results: Codable {
        var load_s: Double             // engine create time (cold GPU spec may land here or in run 1)
        var runs: [Run]
    }
    struct Timestamps: Codable {
        var started: String            // ISO8601 UTC
        var finished: String
    }

    var device: Device
    var app: AppInfo
    var model: Model
    var benchProtocol: BenchProtocol
    var environment: Environment
    var results: Results
    var timestamps: Timestamps

    enum CodingKeys: String, CodingKey {
        case schema_version, kind, device, app, model
        case benchProtocol = "protocol"
        case environment, results, timestamps
    }
}

// MARK: - Runner

@MainActor
final class BenchRunner: ObservableObject {
    @Published var lines: [String] = []
    @Published var running = false
    @Published var blob: BenchBlob?
    @Published var blobJSON = ""

    // Fixed protocol (v1). Changing any of these requires a new protocol name —
    // the aggregator rejects blobs whose protocol block doesn't match exactly.
    static let protocolName = "pb-random-v1"
    static let promptTokens = 128
    static let maxTokens = 256
    static let promptSeed: UInt64 = 0
    static let coldRuns = 1
    static let warmRuns = 3

    func add(_ line: String) {
        print(line)
        lines.append(line)
    }

    func run(model: BenchModel) async {
        guard !running else { return }
        running = true
        blob = nil
        blobJSON = ""
        lines = []
        defer { running = false }

        guard model.isInstalled else {
            add("ERROR \(model.id): bundle not installed")
            return
        }
        // S=1 prefill contract (same guard as the chat backends). Recorded, and
        // the aggregator rejects blobs where this isn't 1.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let chunkThreshold = Int(String(cString: getenv("COREAI_CHUNK_THRESHOLD"))) ?? -1

        UIDevice.current.isBatteryMonitoringEnabled = true
        let iso = ISO8601DateFormatter()
        let started = iso.string(from: Date())
        let deviceMeta = Self.deviceMeta()
        add("device: \(deviceMeta.model_identifier) · \(deviceMeta.os) (\(deviceMeta.os_build)) · \(String(format: "%.0f", deviceMeta.memory_gb)) GB")
        add("model: \(model.id) (\(model.bundleName))")

        let thermalBefore = Self.thermalString(ProcessInfo.processInfo.thermalState)
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let batteryLevel = Double(UIDevice.current.batteryLevel)
        let batteryState = Self.batteryString(UIDevice.current.batteryState)
        let availMB = Int(Double(os_proc_available_memory()) / 1e6)
        let freeDiskGB = Self.freeDiskGB()
        add(String(format: "env: thermal %@ · battery %.0f%% %@ · LPM %@ · avail mem %d MB",
                   thermalBefore, max(0, batteryLevel) * 100, batteryState,
                   lowPower ? "ON" : "off", availMB))

        // Network metadata BEFORE the timed section (never during).
        let revision = await Self.fetchRevision(repo: model.repo)
        add("hf: \(model.repo) @ \(revision)")

        do {
            // Engine create (timed). Fresh engine per bench run — no reuse from the chat tab.
            let bundle = try LanguageBundle(at: model.bundleDir)
            let bundleKind = Self.bundleKind(in: model.bundleDir)
            let config = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            add("creating engine (cold GPU specialization may land here or in run 1)…")
            let t0 = SuspendingClock.now
            let engine = try await EngineFactory.createEngine(
                config: try JSONEncoder().encode(config),
                modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
                options: EngineOptions()
            )
            let loadS = Self.seconds(since: t0)
            add(String(format: "engine ready in %.1f s", loadS))

            // Fixed prompt: splitmix64(seed 0) over the model's vocab — identical to
            // PipelinedBench / llm-benchmark, so community rows are comparable to the
            // precision-bench numbers while staying tokenizer-agnostic.
            let prompt = Self.randomPrompt(vocabSize: bundle.vocabSize,
                                           count: Self.promptTokens, seed: Self.promptSeed)
            var runs: [BenchBlob.Run] = []
            for i in 0..<(Self.coldRuns + Self.warmRuns) {
                let kind = i < Self.coldRuns ? "cold" : "warm"
                let r = try await Self.speedTrial(engine: engine, prompt: prompt,
                                                  maxTokens: Self.maxTokens)
                let run = BenchBlob.Run(
                    kind: kind,
                    prefill_tok_s: r.prefillS > 0 ? Double(prompt.count) / r.prefillS : 0,
                    decode_tok_s: r.decodeS > 0 ? Double(max(0, r.genTokens - 1)) / r.decodeS : 0,
                    prefill_s: r.prefillS,
                    decode_s: r.decodeS,
                    gen_tokens: r.genTokens,
                    thermal_state_end: Self.thermalString(ProcessInfo.processInfo.thermalState))
                runs.append(run)
                add(String(format: "RUN %d (%@) prefill=%.1f decode=%.1f tok/s · thermal %@",
                           i + 1, kind, run.prefill_tok_s, run.decode_tok_s, run.thermal_state_end))
            }

            let thermalAfter = Self.thermalString(ProcessInfo.processInfo.thermalState)
            let finished = iso.string(from: Date())
            let info = Bundle.main.infoDictionary
            let result = BenchBlob(
                schema_version: 1,
                kind: "coreai-community-bench",
                device: deviceMeta,
                app: BenchBlob.AppInfo(
                    version: info?["CFBundleShortVersionString"] as? String ?? "unknown",
                    build: info?["CFBundleVersion"] as? String ?? "unknown"),
                model: BenchBlob.Model(
                    id: model.id, hf_repo: model.repo, hf_revision: revision,
                    bundle: model.bundleName, bundle_kind: bundleKind),
                benchProtocol: BenchBlob.BenchProtocol(
                    name: Self.protocolName,
                    prompt_tokens: Self.promptTokens, max_tokens: Self.maxTokens,
                    prompt_seed: Int(Self.promptSeed), temperature: 0,
                    chunk_threshold: chunkThreshold,
                    cold_runs: Self.coldRuns, warm_runs: Self.warmRuns),
                environment: BenchBlob.Environment(
                    thermal_state_before: thermalBefore, thermal_state_after: thermalAfter,
                    low_power_mode: lowPower,
                    battery_level: (batteryLevel * 100).rounded() / 100,
                    battery_state: batteryState,
                    available_memory_mb: availMB, free_disk_gb: freeDiskGB),
                results: BenchBlob.Results(load_s: (loadS * 100).rounded() / 100, runs: runs),
                timestamps: BenchBlob.Timestamps(started: started, finished: finished))

            let enc = JSONEncoder()
            enc.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
            let json = String(data: try enc.encode(result), encoding: .utf8) ?? "{}"
            blob = result
            blobJSON = json

            let warmDecodes = runs.filter { $0.kind == "warm" }.map(\.decode_tok_s).sorted()
            let med = warmDecodes.isEmpty ? 0 : warmDecodes[warmDecodes.count / 2]
            add(String(format: "STATS BENCH model=%@ load=%.1fs warm_decode_med=%.1f tok/s",
                       model.id, loadS, med))
            try? Self.persist(json: json, modelID: model.id)
        } catch {
            let ns = error as NSError
            add("ERROR \(error.localizedDescription)")
            add("ERROR-DETAIL domain=\(ns.domain) code=\(ns.code)")
        }
    }

    // MARK: speed trial (mirrors PipelinedBench.speedTrial / llm-benchmark runTrial)

    struct TrialResult { let prefillS: Double; let decodeS: Double; let genTokens: Int }

    private static func speedTrial(
        engine: any InferenceEngine, prompt: [Int32], maxTokens: Int
    ) async throws -> TrialResult {
        try? await Task.sleep(for: .milliseconds(50))
        try await engine.reset()
        let start = SuspendingClock.now
        let stream = try engine.generate(
            with: prompt,
            samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens)
        )
        var promptTime = 0.0
        var genStart = SuspendingClock.now
        var count = 0
        for try await _ in stream {
            if promptTime == 0 {
                promptTime = seconds(since: start)
                genStart = SuspendingClock.now
            }
            count += 1
        }
        return TrialResult(prefillS: promptTime, decodeS: seconds(since: genStart), genTokens: count)
    }

    // splitmix64 — same constants and seed as PipelinedBench / llm-benchmark.
    private static func randomPrompt(vocabSize: Int, count: Int, seed: UInt64) -> [Int32] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var out = [Int32]()
        out.reserveCapacity(count)
        let v = UInt64(vocabSize)
        for _ in 0..<count {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            out.append(Int32(z % v))
        }
        return out
    }

    private static func seconds(since start: SuspendingClock.Instant) -> Double {
        let d = SuspendingClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    // MARK: metadata capture

    private static func deviceMeta() -> BenchBlob.Device {
        var sys = utsname()
        uname(&sys)
        let machine = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
        }
        let os = "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
        let memGB = (Double(ProcessInfo.processInfo.physicalMemory) / 1e9 * 10).rounded() / 10
        return BenchBlob.Device(model_identifier: machine, os: os,
                                os_build: sysctlString("kern.osversion"), memory_gb: memGB)
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname(name, &buf, &size, nil, 0)
        return String(cString: buf)
    }

    static func thermalString(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    private static func batteryString(_ s: UIDevice.BatteryState) -> String {
        switch s {
        case .unplugged: "unplugged"
        case .charging: "charging"
        case .full: "full"
        default: "unknown"
        }
    }

    private static func freeDiskGB() -> Double {
        let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())
        let free = (attrs?[.systemFreeSize] as? Int64) ?? 0
        return (Double(free) / 1e9 * 10).rounded() / 10
    }

    private static func bundleKind(in dir: URL) -> String {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        if names.contains(where: { $0.hasSuffix(".aimodelc") }) { return "aimodelc" }
        if names.contains(where: { $0.hasSuffix(".aimodel") }) { return "aimodel" }
        return "unknown"
    }

    // Repo main revision at bench time — best-effort (5 s), "unknown" offline.
    private static func fetchRevision(repo: String) async -> String {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)") else { return "unknown" }
        var req = URLRequest(url: url)
        req.timeoutInterval = 5
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sha = obj["sha"] as? String else { return "unknown" }
        return sha
    }

    // MARK: persistence + submission

    // Keep a copy under Documents/bench/ (auditable later; devicectl-copyable in CI).
    private static func persist(json: String, modelID: String) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("bench")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        df.timeZone = TimeZone(identifier: "UTC")
        let url = dir.appendingPathComponent("bench_\(modelID)_\(df.string(from: Date())).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
    }

    /// New-issue URL with the blob prefilled into the `blob` field of the zoo's
    /// bench-result issue form. nil when the encoded blob would blow the URL limit —
    /// callers fall back to the bare template (blob goes via the clipboard).
    static func submissionURL(blob: BenchBlob, blobJSON: String) -> URL? {
        var comps = URLComponents(string: "https://github.com/john-rocky/coreai-model-zoo/issues/new")!
        comps.queryItems = [
            URLQueryItem(name: "template", value: "bench-result.yml"),
            URLQueryItem(name: "title", value: "[bench] \(blob.device.model_identifier) · \(blob.model.id)"),
            URLQueryItem(name: "blob", value: blobJSON),
        ]
        guard let url = comps.url, url.absoluteString.count < 7500 else { return nil }
        return url
    }

    static let templateURL = URL(string:
        "https://github.com/john-rocky/coreai-model-zoo/issues/new?template=bench-result.yml")!

    // MARK: headless (BENCH_AUTO=<model id | all>) — the devicectl/DoD entrypoint.
    // Downloads the bundle when missing, runs the bench, prints the blob between
    // BENCH-BLOB-BEGIN/END markers, and leaves a copy in Documents/bench/.
    static func headless(modelID: String) async {
        let targets = modelID == "all"
            ? BenchModel.catalog
            : BenchModel.catalog.filter { $0.id == modelID }
        guard !targets.isEmpty else {
            print("ERROR BENCH_AUTO: unknown model id \(modelID) — have: \(BenchModel.catalog.map(\.id).joined(separator: ", "))")
            return
        }
        for model in targets {
            if !model.isInstalled {
                print("[bench] downloading \(model.id) (~\(model.approxDownloadGB) GB)…")
                let dl = ModelDownloader()
                let probe = Task { @MainActor in
                    while !Task.isCancelled {
                        try? await Task.sleep(nanoseconds: 6_000_000_000)
                        print("[bench] dl \(Int(dl.fraction * 100))% \(dl.detail)")
                    }
                }
                await dl.fetch(repo: "https://huggingface.co/" + model.repo,
                               items: [ModelDownloader.Item(remote: model.remotePath,
                                                            local: model.bundleName)],
                               into: FileManager.default.urls(for: .documentDirectory,
                                                              in: .userDomainMask)[0]
                                   .appendingPathComponent("models"))
                probe.cancel()
                guard dl.phase == .done else {
                    print("ERROR BENCH_AUTO: download failed for \(model.id): \(dl.phase)")
                    continue
                }
            }
            let runner = BenchRunner()
            await runner.run(model: model)
            if !runner.blobJSON.isEmpty {
                print("BENCH-BLOB-BEGIN \(model.id)")
                print(runner.blobJSON)
                print("BENCH-BLOB-END \(model.id)")
            }
        }
    }
}
