// coreai-audio — on-device audio understanding (Qwen2.5-Omni Thinker on Core AI), built on CoreAIKit.

import SwiftUI

@main
struct CoreAIAudioApp: App {
    init() {
        if ProcessInfo.processInfo.environment["KOKORO_SELFTEST"] != nil {
            let sem = DispatchSemaphore(value: 0)
            Task.detached { await runKokoroSelfTest(); sem.signal() }
            sem.wait()
            exit(0)
        }
    }

    var body: some Scene {
        WindowGroup("coreai-audio") {
            TabView {
                ContentView()
                    .tabItem { Label("Understand", systemImage: "ear") }
                KokoroView()
                    .tabItem { Label("Speak", systemImage: "speaker.wave.2") }
            }
        }
        #if os(macOS)
            .defaultSize(width: 560, height: 520)
        #endif
    }
}
