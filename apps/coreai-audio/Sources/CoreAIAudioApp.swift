// coreai-audio — on-device audio understanding (Qwen2.5-Omni Thinker on Core AI), built on CoreAIKit.

import SwiftUI

@main
struct CoreAIAudioApp: App {
    init() {
        // Headless ASR self-test: kicked from init() (not .task) so it runs without the GUI window
        // appearing — reliable for a CLI-launched verification run.
        if ProcessInfo.processInfo.environment["ASR_SELFTEST"] != nil {
            Task.detached { await runASRSelfTest() }
        }
        // Same init()-launch (not .task) as ASR — reliable for a headless CLI verification run.
        if ProcessInfo.processInfo.environment["VOXCPM_SELFTEST"] != nil {
            Task.detached { await runVoxCPMSelfTest() }
        }
        if ProcessInfo.processInfo.environment["VOXCPM2_SELFTEST"] != nil {
            Task.detached { await runVoxCPM2SelfTest() }
        }
        if ProcessInfo.processInfo.environment["PARAKEET_SELFTEST"] != nil {
            Task.detached { await runParakeetSelfTest() }
        }
    }

    var body: some Scene {
        WindowGroup("coreai-audio") {
            TabView {
                ContentView()
                    .tabItem { Label("Understand", systemImage: "ear") }
                TranscribeView()
                    .tabItem { Label("Transcribe", systemImage: "text.bubble") }
                KokoroView()
                    .tabItem { Label("Speak", systemImage: "speaker.wave.2") }
                VoxCPMView()
                    .tabItem { Label("Voice", systemImage: "waveform") }
                VoxCPM2View()
                    .tabItem { Label("Voice 2B", systemImage: "waveform.badge.plus") }
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
