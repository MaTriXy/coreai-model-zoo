// Recorder — minimal mic capture to a 16 kHz mono WAV via AVAudioRecorder (not AVAudioEngine
// taps, which are crash-prone with permission/dispatch ordering). The recorded file is fed
// to the transcribe engine like any picked audio file.

import AVFoundation
import Foundation

@MainActor
final class Recorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var lastURL: URL?

    private var recorder: AVAudioRecorder?

    func requestPermissionAndStart() {
        #if os(iOS)
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor in if granted { self?.start() } }
        }
        #else
        start()
        #endif
    }

    func start() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, mode: .default)
        try? session.setActive(true)
        #endif
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreai_recording.wav")
        try? FileManager.default.removeItem(at: url)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.record()
            recorder = r
            isRecording = true
        } catch {
            isRecording = false
        }
    }

    /// Stop and return the recorded file URL.
    @discardableResult
    func stop() -> URL? {
        recorder?.stop()
        let url = recorder?.url
        recorder = nil
        isRecording = false
        lastURL = url
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false)
        #endif
        return url
    }
}
