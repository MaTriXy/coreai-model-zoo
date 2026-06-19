// AudioArchitecture.swift — fixed-shape geometry for the Qwen2.5-Omni Thinker audio decoder + its
// paired Whisper-style audio encoder. The decoder keeps a plain Qwen2.5 text path and rides the
// audio state on ONE static graph input, `audio_embeds [maxAudioTokens, hidden]`. Simpler than the
// VL rider: TMRoPE collapses to 1-D for audio+text, so positions are the engine's native `arange`
// (no rope-shift inputs). The encoder is a fixed K-chunk graph:
//   input_features [1, melBins, chunkMel*chunks] + attn_bias [chunks,1,1,chunkMel/2]
//     -> audio_embeds [chunks*50, hidden]
// (host zero-pads the mel to whole chunks, trims the output to the clip's real audio-token N).

import Foundation

struct AudioArchitecture: Sendable {
    let vocab: Int32          // text vocab; audio tokens are vocab+slot
    let hidden: Int           // decoder / encoder width (Qwen2.5-Omni-3B = 2048)
    let maxAudioTokens: Int   // audio_embeds row count = chunks*50 (longest clip)
    let chunks: Int           // fixed mel-chunk count baked into the encoder (K=15 ≈ 30 s)
    let chunkMel: Int         // mel frames per chunk before the stride-2 CNN (200)
    let melBins: Int          // Whisper-large-v3 mel filterbank bins (128)

    // Special tokens (Qwen ChatML + Qwen2.5-Omni audio framing). audioPad must be one token.
    let imStart = "<|im_start|>"
    let imEnd = "<|im_end|>"
    let audioBos = "<|audio_bos|>"
    let audioEos = "<|audio_eos|>"
    let audioPad = "<|AUDIO|>"

    var headsFrames: Int { chunkMel / 2 }           // post-CNN frames per chunk (100)
    var melFrames: Int { chunkMel * chunks }        // total mel frames (3000 @ K=15)
    var inputFeaturesCount: Int { melBins * melFrames }
    var attnBiasCount: Int { chunks * headsFrames }
    var audioEmbedsCount: Int { maxAudioTokens * hidden }

    /// Audio-token count for `melValidFrames` real mel frames (conv1+conv2/2+avg_pool/2).
    func audioTokenCount(melValidFrames: Int) -> Int {
        melValidFrames <= 0 ? 0 : ((melValidFrames - 1) / 2 + 1 - 2) / 2 + 1
    }

    /// Pack a Whisper log-mel [melBins, frames] into the encoder's fixed inputs (mirrors the
    /// python host contract): input_features [1, melBins, melFrames] (left-aligned, zero-padded to
    /// K chunks) + attn_bias [chunks,1,1,headsFrames] (−30000 on each chunk's padded post-CNN
    /// frames) + the clip's audio-token count N.
    func encoderInputs(fromMel mel: [Float], frames: Int)
        -> (inputFeatures: [Float16], attnBias: [Float16], audioTokenCount: Int)
    {
        let neg: Float16 = -30000
        var feats = [Float16](repeating: 0, count: inputFeaturesCount)
        let copyT = min(frames, melFrames)
        for c in 0..<melBins {
            let dst = c * melFrames, src = c * frames
            for t in 0..<copyT { feats[dst + t] = Float16(mel[src + t]) }
        }
        var bias = [Float16](repeating: 0, count: attnBiasCount)
        for kc in 0..<chunks {
            let melInChunk = max(0, min(chunkMel, frames - kc * chunkMel))
            let valid = melInChunk == 0 ? 0 : (melInChunk - 1) / 2 + 1
            if valid < headsFrames {
                for j in valid..<headsFrames { bias[kc * headsFrames + j] = neg }
            }
        }
        return (feats, bias, audioTokenCount(melValidFrames: frames))
    }

    /// Qwen2.5-Omni-3B: 2048-wide, K=15 encoder (≈30 s, 750 audio-embed rows).
    static let qwen2_5Omni3B = AudioArchitecture(
        vocab: 151_936, hidden: 2048, maxAudioTokens: 750, chunks: 15, chunkMel: 200, melBins: 128)
}
