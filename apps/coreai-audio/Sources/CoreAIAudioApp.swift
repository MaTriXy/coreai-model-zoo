// coreai-audio — on-device audio understanding (Qwen2.5-Omni Thinker on Core AI), built on CoreAIKit.

import SwiftUI

@main
struct CoreAIAudioApp: App {
    var body: some Scene {
        WindowGroup("coreai-audio") {
            TabView {
                ContentView()
                    .tabItem { Label("Understand", systemImage: "ear") }
                KokoroView()
                    .tabItem { Label("Speak", systemImage: "speaker.wave.2") }
            }
            // Non-blocking self-test (KOKORO_SELFTEST=1): the iOS launch watchdog
            // kills any main-thread block, so run it as a normal async task.
            .task {
                if ProcessInfo.processInfo.environment["KOKORO_SELFTEST"] != nil {
                    await runKokoroSelfTest()
                }
            }
        }
        #if os(macOS)
            .defaultSize(width: 560, height: 520)
        #endif
    }
}
