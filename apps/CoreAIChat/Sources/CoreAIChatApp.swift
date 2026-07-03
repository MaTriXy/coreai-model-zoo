// CoreAIChat — on-device Core AI LLM chat for iPhone (iOS 27).
// Single app, one picker, seven engines: gemma 4 E2B GPU (kernel monolith) / gemma 4 E2B ANE
// (host-cache chunks) / gemma 4 E2B ⚡pipelined (int4lin, PLE table as a static graph input —
// 30.3 tok/s decode settled vs 22-24 for the monolith) / Qwen3.5-0.8B ⚡pipelined
// (int8hu head, 69.7-74.0 tok/s runner on iPhone 17 Pro) / Qwen3.5-2B ⚡pipelined /
// LFM2.5-1.2B ⚡pipelined (38.0-39.6 tok/s) / Granite-4.0-H-1B ⚡pipelined — the pipelined
// set rides Apple's coreai-pipelined engine on decode-only loop-free bundles.

import SwiftUI

@main
struct CoreAIChatApp: App {
    @StateObject private var engine = Gemma4ChatEngine()
    @State private var tab = 0
    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                chatTab
                    .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                    .tag(0)
                BenchView(chatEngine: engine)
                    .tabItem { Label("Bench", systemImage: "speedometer") }
                    .tag(1)
            }
            .onChange(of: tab) { _, t in
                // A bench run frees the chat engine for memory headroom; coming back to
                // the Chat tab reloads it lazily (files are still on disk).
                if t == 0, !engine.ready, !engine.loading, engine.installedForCurrentMode {
                    Task { await engine.load() }
                }
            }
        }
    }

    private var chatTab: some View {
            ChatView(engine: engine)
                .task {
                    // GEMMA_CLEAR_SPEC_CACHE=1: wipe the in-container GPU
                    // specialization cache before anything loads. The recovery
                    // for the partial-e-cache trap: an out-of-disk cold
                    // specialization leaves a partial entry that fails every
                    // later engine create with NSPOSIXErrorDomain code=2 —
                    // without this hook the only fix is uninstalling the app
                    // (losing every sideloaded bundle). Costs one cold
                    // re-specialization per model.
                    if ProcessInfo.processInfo.environment["GEMMA_CLEAR_SPEC_CACHE"] == "1" {
                        let caches = FileManager.default.urls(
                            for: .cachesDirectory, in: .userDomainMask)[0]
                        for name in ["coreai-cache", "com.apple.MetalPerformanceShadersGraph"] {
                            let dir = caches.appendingPathComponent(name)
                            if FileManager.default.fileExists(atPath: dir.path) {
                                try? FileManager.default.removeItem(at: dir)
                                print("[chat] cleared spec cache: \(name)")
                            }
                        }
                    }
                    // BENCH_AUTO=<model id | all>: headless community-bench run (downloads the
                    // bundle when missing, prints the blob between BENCH-BLOB-BEGIN/END).
                    if let benchID = ProcessInfo.processInfo.environment["BENCH_AUTO"] {
                        await BenchRunner.headless(modelID: benchID)
                        return
                    }
                    // De-risk path: GEMMA_SLICE_TEST=1 runs the ISOLATED dual-KV ANE slice self-test
                    // (does the gemma4 iOS port lower+run on the ANE?) and skips the Phase-1 engine.
                    if ProcessInfo.processInfo.environment["GEMMA_SLICE_TEST"] == "1" {
                        await SliceANETest.run()
                        return
                    }
                    // BITVLA_TEST=1: zoo's first VLA — 1.58-bit BitVLA image+instruction -> 7-DoF on the GPU.
                    if ProcessInfo.processInfo.environment["BITVLA_TEST"] == "1" {
                        await BitVLATest.run()
                        return
                    }
                    // Session B's host-cache core (no state / no in-graph indexed write) on device.
                    if ProcessInfo.processInfo.environment["GEMMA_HOSTCACHE_TEST"] == "1" {
                        await HostCacheTest.run()
                        return
                    }
                    // CHUNKED host-cache ANE engine: chain the 6 chunks (host-managed KV) -> head -> argmax.
                    // GEMMA_CHUNK_VERIFY=1 = device 8/8 vs the oracle; else generate + tok/s.
                    if ProcessInfo.processInfo.environment["GEMMA_CHUNK_TEST"] == "1" {
                        await Gemma4ChunkEngine.run()
                        return
                    }
                    // GPU MONOLITH engine (fused-int8 Metal-kernel FFN core): gather -> 1 core -> head.
                    if ProcessInfo.processInfo.environment["GEMMA_MONO_TEST"] == "1" {
                        await Gemma4MonolithEngine.run()
                        return
                    }
                    if ProcessInfo.processInfo.environment["GEMMA_DL_TEST"] == "1" {
                        await engine.downloadSelfTest(); return
                    }
                    await engine.load()
                    // Headless self-test for `devicectl process launch --console
                    // --environment-variables '{"GEMMA_PROMPT":"..."}'` — prints result + tok/s.
                    await engine.autoTestIfRequested()
                }
    }
}
