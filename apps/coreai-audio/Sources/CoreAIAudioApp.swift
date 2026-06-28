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
                VoxCPMView()
                    .tabItem { Label("Voice", systemImage: "waveform") }
                VoxCPM2View()
                    .tabItem { Label("Voice 2B", systemImage: "waveform.badge.plus") }
            }
            // Non-blocking self-tests (the iOS launch watchdog kills any main-thread block).
            .task {
                if ProcessInfo.processInfo.environment["KOKORO_SELFTEST"] != nil {
                    await runKokoroSelfTest()
                }
                if ProcessInfo.processInfo.environment["VOXCPM_SELFTEST"] != nil {
                    await runVoxCPMSelfTest()
                }
                if ProcessInfo.processInfo.environment["VOXCPM2_SELFTEST"] != nil {
                    await runVoxCPM2SelfTest()
                }
            }
        }
        #if os(macOS)
            .defaultSize(width: 560, height: 520)
        #endif
    }
}
