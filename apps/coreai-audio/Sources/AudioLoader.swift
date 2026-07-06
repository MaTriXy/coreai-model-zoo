// AudioLoader.swift — decode any audio file to 16 kHz mono Float (the model's input rate), and a
// synthetic demo clip so the app always has something to ask about without a file.

import AVFoundation
import Foundation

enum AudioLoader {
    /// Decode an audio file to 16 kHz mono `[Float]` (resampling via `AVAudioConverter`).
    static func load16kMono(_ url: URL) -> [Float]? {
        guard url.startAccessingSecurityScopedResource() || true else { return nil }
        defer { url.stopAccessingSecurityScopedResource() }
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let src = file.processingFormat
        guard
            let dst = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: src, to: dst),
            let srcBuf = AVAudioPCMBuffer(
                pcmFormat: src, frameCapacity: AVAudioFrameCount(file.length))
        else { return nil }
        do { try file.read(into: srcBuf) } catch { return nil }

        let outCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * 16000 / src.sampleRate) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outCapacity) else {
            return nil
        }
        var fed = false
        var err: NSError?
        converter.convert(to: dstBuf, error: &err) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return srcBuf
        }
        guard err == nil, let channel = dstBuf.floatChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(dstBuf.frameLength)))
    }

    /// Decode an audio file to 44.1 kHz stereo `[[Float]]` (2 channels) for source separation.
    static func load44kStereo(_ url: URL) -> [[Float]]? {
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let src = file.processingFormat
        guard
            let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false),
            let converter = AVAudioConverter(from: src, to: dst),
            let srcBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(file.length))
        else { return nil }
        do { try file.read(into: srcBuf) } catch { return nil }
        let outCap = AVAudioFrameCount(Double(srcBuf.frameLength) * 44_100 / src.sampleRate) + 4096
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outCap) else { return nil }
        var fed = false; var err: NSError?
        converter.convert(to: dstBuf, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return srcBuf
        }
        guard err == nil, let ch = dstBuf.floatChannelData else { return nil }
        let n = Int(dstBuf.frameLength)
        let l = Array(UnsafeBufferPointer(start: ch[0], count: n))
        let r = dst.channelCount > 1 ? Array(UnsafeBufferPointer(start: ch[1], count: n)) : l
        return [l, r]
    }

    /// 4 s of deterministic white noise (the gated demo clip — answered "hissing sound").
    static func demoNoise(seconds: Int = 4) -> [Float] {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Double(state >> 11) / Double(1 << 53)) * 2 - 1
        }
        return (0..<(16000 * seconds)).map { _ in next() * 0.9 }
    }
}
