// ContinueInAppIntent.swift — the optional "Continue in app" / "More results" surface.
//
// Conforms to Apple's `semanticContentSearch` schema via `@AppIntent(schema:)`. When the user
// taps "more results" in Visual Intelligence, the app opens and answers the captured frame with
// MiniCPM-V in the FOREGROUND — the full app memory budget, sidestepping the background VI
// launch's tighter ceiling. This is the robust default path (works on a cold cache too).
//
// Deliberately ISOLATED in its own file: the `@AppIntent(schema: .visualIntelligence.semanticContentSearch)`
// macro is the least-stable part against the 27 beta SDK, so if it fails to build you can delete
// just this file and the core Visual Intelligence integration (query + entity + OpenIntent) stands.

import AppIntents
import Foundation
import VisualIntelligence

@available(iOS 26.0, *)
@AppIntent(schema: .visualIntelligence.semanticContentSearch)
struct ContinueVisualSearchInAppIntent {
    static let title: LocalizedStringResource = "Ask MiniCPM-V about this"
    static let openAppWhenRun: Bool = true

    var semanticContent: SemanticContentDescriptor

    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        // Hand the capture to the foreground app as JPEG `Data` (Sendable). A touch larger than
        // the VI row thumbnail so the foreground answer has more detail to work with.
        var data: Data?
        if let pixelBuffer = semanticContent.pixelBuffer,
            let cgImage = MiniCPMVIEngine.cgImage(from: pixelBuffer)
        {
            data = MiniCPMVIEngine.jpeg(from: cgImage, maxSide: 768)
        }
        await router.continueInApp(imageData: data)
        return .result()
    }
}
