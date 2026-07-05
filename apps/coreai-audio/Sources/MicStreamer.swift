// MicStreamer — continuous microphone capture for live streaming ASR: an AVAudioEngine input tap
// converted to 16 kHz mono Float packets, delivered through an AsyncStream. Unlike MicRecorder
// (record-then-decode, used by the batch engines), packets arrive while the user is speaking.
// The tap callback only converts + yields — all model work happens on the consumer task.

import AVFoundation
import Foundation

final class MicStreamer: @unchecked Sendable {
    enum MicError: LocalizedError {
        case denied, noInput
        var errorDescription: String? {
            switch self {
            case .denied: return "Microphone access was denied (enable it in Settings)."
            case .noInput: return "No microphone input is available."
            }
        }
    }

    private var engine: AVAudioEngine?
    private var converter: AVAudioConverter?
    private var continuation: AsyncStream<[Float]>.Continuation?

    var isStreaming: Bool { engine?.isRunning ?? false }

    /// Request permission and start the tap. Returns a stream of 16 kHz mono packets
    /// (~50-100 ms each, hardware-dependent). Finishes when `stop()` is called.
    func start() async throws -> AsyncStream<[Float]> {
        #if os(iOS)
            let granted = await AVAudioApplication.requestRecordPermission()
            guard granted else { throw MicError.denied }
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers])
            try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let inFormat = input.inputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else { throw MicError.noInput }
        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw MicError.noInput }

        let (stream, continuation) = AsyncStream.makeStream(of: [Float].self,
                                                            bufferingPolicy: .unbounded)
        self.engine = engine
        self.converter = converter
        self.continuation = continuation

        input.installTap(onBus: 0, bufferSize: 1600, format: inFormat) { [weak self] buffer, _ in
            guard let self, let converter = self.converter else { return }
            let ratio = 16000.0 / inFormat.sampleRate
            let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
            var fed = false
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                if fed {
                    status.pointee = .noDataNow
                    return nil
                }
                fed = true
                status.pointee = .haveData
                return buffer
            }
            guard err == nil, out.frameLength > 0, let ch = out.floatChannelData else { return }
            let packet = Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
            self.continuation?.yield(packet)
        }

        engine.prepare()
        try engine.start()
        return stream
    }

    /// Stop the tap and finish the stream (the consumer's `for await` loop ends).
    func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        converter = nil
        continuation?.finish()
        continuation = nil
        #if os(iOS)
            try? AVAudioSession.sharedInstance().setActive(false)
        #endif
    }
}
