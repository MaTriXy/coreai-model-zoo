// CoreAISegment — a minimal cross-platform (macOS + iOS) text-prompt image
// segmentation app for the SAM 3 Core AI bundle, running on Apple's official
// CoreAIImageSegmenter runtime (apple/coreai-models). Pick an image, type what to
// segment ("cat", "the red car"), and the model returns instance masks.

import SwiftUI

@main
struct CoreAISegmentApp: App {
    var body: some Scene {
        WindowGroup {
            #if os(macOS)
            ContentView()
                .frame(minWidth: 860, minHeight: 620)
            #else
            ContentView()
            #endif
        }
        #if os(macOS)
        .windowResizability(.contentMinSize)
        #endif
    }
}
