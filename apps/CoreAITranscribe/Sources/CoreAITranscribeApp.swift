// CoreAITranscribe — a minimal cross-platform (macOS + iOS) speech-to-text app for the
// Whisper large-v3-turbo Core AI bundle, running on Apple's stock Core AI runtime. Pick an
// audio file or record from the mic; the model transcribes on-device.

import SwiftUI

@main
struct CoreAITranscribeApp: App {
    var body: some Scene {
        WindowGroup {
            #if os(macOS)
            ContentView().frame(minWidth: 560, minHeight: 520)
            #else
            ContentView()
            #endif
        }
        #if os(macOS)
        .windowResizability(.contentMinSize)
        #endif
    }
}
