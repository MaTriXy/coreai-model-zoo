// AudioRecorder — capture microphone audio and return it as 16 kHz mono Float (the model's input
// rate). Taps the AVAudioEngine input node and resamples each buffer through an AVAudioConverter;
// also decodes a chosen file and synthesizes a demo clip.

import AVFoundation
import Foundation

@MainActor
final class AudioRecorder: ObservableObject {
    @Published private(set) var isRecording = false

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private var samples: [Float] = []

    /// Begin capturing. The first call prompts for microphone permission.
    func start() throws {
        guard !isRecording else { return }
        samples.removeAll(keepingCapacity: true)
        #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [])
            try session.setActive(true)
        #endif
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inFormat, to: target)
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.appendConverted(buffer)
        }
        engine.prepare()
        try engine.start()
        isRecording = true
    }

    /// Stop capturing and return the recorded 16 kHz mono samples.
    func stop() -> [Float] {
        guard isRecording else { return samples }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        #if os(iOS)
            try? AVAudioSession.sharedInstance().setActive(false)
        #endif
        isRecording = false
        return samples
    }

    private nonisolated func appendConverted(_ buffer: AVAudioPCMBuffer) {
        // Hop off the realtime tap thread to touch main-actor state.
        let pcm = Self.resample(buffer, to: target)
        Task { @MainActor [weak self] in self?.samples.append(contentsOf: pcm) }
    }

    private nonisolated static func resample(_ buffer: AVAudioPCMBuffer, to target: AVAudioFormat) -> [Float] {
        guard let converter = AVAudioConverter(from: buffer.format, to: target) else { return [] }
        let ratio = target.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 256
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return [] }
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        guard err == nil, let ch = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    // MARK: - File + demo

    /// Decode any audio file to 16 kHz mono.
    static func load16kMono(_ url: URL) -> [Float]? {
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        guard let buf = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { return nil }
        do { try file.read(into: buf) } catch { return nil }
        let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        return resample(buf, to: target)
    }

    /// 4 s of deterministic white noise (answered "hissing sound").
    static func demoNoise(seconds: Int = 4) -> [Float] {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Double(state >> 11) / Double(1 << 53)) * 2 - 1
        }
        return (0..<(16000 * seconds)).map { _ in next() * 0.9 }
    }
}
