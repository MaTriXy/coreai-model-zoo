// coreai-video — on-device video action recognition (V-JEPA 2 on Core AI).
import SwiftUI

@main
struct CoreAIVideoApp: App {
    init() {
        // Headless self-test (ACTION_SELFTEST=1): init()-launched so it runs reliably from the CLI.
        if ProcessInfo.processInfo.environment["ACTION_SELFTEST"] != nil {
            Task.detached { await runActionSelfTest() }
        }
    }

    var body: some Scene {
        WindowGroup("coreai-video") {
            ActionView()
        }
        #if os(macOS)
            .defaultSize(width: 640, height: 520)
        #endif
    }
}
