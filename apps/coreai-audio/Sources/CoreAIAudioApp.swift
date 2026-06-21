// coreai-audio — on-device audio understanding (Qwen2.5-Omni Thinker on Core AI), built on CoreAIKit.

import SwiftUI

@main
struct CoreAIAudioApp: App {
    var body: some Scene {
        WindowGroup("coreai-audio") {
            ContentView()
        }
        #if os(macOS)
            .defaultSize(width: 560, height: 520)
        #endif
    }
}
