// VisualAnswerEntity.swift — the one App Intents entity the system Visual Intelligence UI
// renders for this app: a short answer about the captured frame. The system has no idea a
// converted MiniCPM-V-4.6 model produced it — an entity is all it sees. The entity needs an
// `OpenIntent` (see VisualSearchIntents.swift) or the app will not surface in Visual
// Intelligence at all.

import AppIntents
import Foundation

/// One on-device answer about the captured frame.
///   • `onDevice == true`  — the answer was already produced by MiniCPM-V (flag A path).
///   • `onDevice == false` — a "tap to ask on-device" teaser; opening it runs the model in the
///     foreground with the full app memory budget (B path).
@available(iOS 27.0, *)
struct VisualAnswerEntity: AppEntity {
    var id: String
    var answer: String
    var onDevice: Bool
    var thumbnail: Data?

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "On-device Answer"

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(
            title: "\(answer)",
            subtitle: onDevice
                ? "MiniCPM-V 4.6 · on-device, no cloud" : "Tap to answer on-device — no cloud",
            image: thumbnail.map { .init(data: $0) })
    }

    static let defaultQuery = VisualAnswerQuery()
}

@available(iOS 27.0, *)
struct VisualAnswerQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [VisualAnswerEntity] {
        // Answers are transient (recomputed per capture) — nothing to rehydrate from an id alone.
        // Return a minimal stand-in so a late lookup still resolves rather than vanishing.
        identifiers.map { VisualAnswerEntity(id: $0, answer: "", onDevice: false, thumbnail: nil) }
    }
}
