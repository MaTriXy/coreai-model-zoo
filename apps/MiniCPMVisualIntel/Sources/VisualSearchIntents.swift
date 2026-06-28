// VisualSearchIntents.swift — the system-facing surface of the app, and it is entirely
// model-agnostic: the system hands us a `SemanticContentDescriptor` (a captured camera frame on
// iPhone, a screenshot elsewhere) and renders whatever `AppEntity`s we return. MiniCPM-V-4.6 runs
// inside `values(for:)`; the system never knows or cares what produced the answer.
//
// Two pieces — both required for the app to appear in Visual Intelligence:
//   1. `IntentValueQuery`  — receives the pixels, returns the answer entity.
//   2. `OpenIntent` for the entity — taps in the VI result list route here. WITHOUT an OpenIntent
//      the app does not surface in Visual Intelligence at all.
// (The optional "Continue in app" schema intent lives in ContinueInAppIntent.swift so it can be
// removed independently if the beta macro misbehaves, leaving the gate intact.)

import AppIntents
import CoreVideo
import Foundation
import VisualIntelligence

// MARK: - 1. The query Visual Intelligence calls with the captured pixels

@available(iOS 27.0, *)
struct VisualSearchValueQuery: IntentValueQuery {
    @Dependency var engine: MiniCPMVIEngine

    func values(for input: SemanticContentDescriptor) async throws -> [VisualAnswerEntity] {
        guard let pixelBuffer = input.pixelBuffer,
            let cgImage = MiniCPMVIEngine.cgImage(from: pixelBuffer)
        else { return [] }

        // CGImage -> Sendable values here, off the main actor, so the @MainActor engine never
        // receives a non-Sendable CGImage.
        let thumbnail = MiniCPMVIEngine.jpeg(from: cgImage)

        // A path (flag ON): run MiniCPM-V right inside the background VI launch. `try?` so a
        // throw or jetsam degrades to the teaser below — the surface never goes empty.
        if MiniCPMVIEngine.runInVisualIntelligence, MiniCPMVIEngine.bundlesPresent {
            let pixels = MiniCPMVIEngine.pixels(from: cgImage)
            if let answer = try? await engine.caption(pixels: pixels),
                !answer.isEmpty
            {
                return [
                    VisualAnswerEntity(
                        id: "answer", answer: answer, onDevice: true, thumbnail: thumbnail)
                ]
            }
        }

        // Default / fallback (B path): a teaser that opens the app to answer with the full
        // foreground memory budget — robust on a cold cache and a tight background budget.
        return [
            VisualAnswerEntity(
                id: "ask", answer: "Ask MiniCPM-V 4.6 about this", onDevice: false,
                thumbnail: thumbnail)
        ]
    }
}

// MARK: - 2. OpenIntent — required, or the app is invisible to Visual Intelligence

@available(iOS 27.0, *)
struct OpenVisualAnswerIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open Answer"

    @Parameter(title: "Answer")
    var target: VisualAnswerEntity

    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        // A tapped on-device answer just shows; a tapped teaser asks the foreground app to run
        // MiniCPM-V on the same capture (full memory budget).
        await router.open(
            answer: target.answer, alreadyAnswered: target.onDevice, thumbnail: target.thumbnail)
        return .result()
    }
}
