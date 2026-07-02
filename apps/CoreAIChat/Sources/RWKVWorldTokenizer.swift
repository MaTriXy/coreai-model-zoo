// RWKV World tokenizer — greedy longest byte-match trie. Community port — NOT an Apple model.
//
// Reads the base64 vocab `rwkv_vocab.tsv` ("<idx>\t<base64(token bytes)>" per line) emitted by
// conversion/rwkv7/prep_vocab.py, which is byte-exact verified against the upstream reference
// RWKV_TOKENIZER (incl. CJK / emoji / symbols). Bundle the .tsv next to the model on device.
//
// The World tokenizer is a raw-byte tokenizer (NOT BPE): encode = greedy longest match over a
// byte trie; decode = concatenate token bytes and UTF-8 decode. Token 0 is the special
// <|rwkv_tokenizer_end_of_text|> (EOS / chat separator) and is not in the vocab file.
import Foundation

final class RWKVWorldTokenizer {
    static let eosToken = 0   // <|rwkv_tokenizer_end_of_text|>

    private final class Node {
        var to: [UInt8: Node] = [:]
        var value: Int?
    }

    private let root = Node()
    private var idx2tok: [Int: [UInt8]] = [:]

    init(vocabURL: URL) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2, let idx = Int(parts[0]),
                  let data = Data(base64Encoded: String(parts[1])) else { continue }
            let bytes = [UInt8](data)
            idx2tok[idx] = bytes
            var u = root
            for ch in bytes {
                if let n = u.to[ch] { u = n } else { let n = Node(); u.to[ch] = n; u = n }
            }
            u.value = idx
        }
    }

    /// UTF-8 bytes -> token ids (greedy longest match, mirrors RWKV_TOKENIZER.encodeBytes).
    func encode(_ s: String) -> [Int] {
        let src = [UInt8](s.utf8)
        let n = src.count
        var out: [Int] = []
        var i = 0
        while i < n {
            var u = root
            var j = i
            var last: (end: Int, tok: Int)?
            while j < n, let nxt = u.to[src[j]] {
                u = nxt
                j += 1
                if let v = u.value { last = (j, v) }
            }
            guard let hit = last else { i += 1; continue }   // every byte is a 1-byte token, so unreachable
            out.append(hit.tok)
            i = hit.end
        }
        return out
    }

    /// Token ids -> string (concatenate token bytes, UTF-8 decode). Robust to multi-byte chars
    /// split across tokens when the full id list is decoded at once.
    func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for t in tokens { if let b = idx2tok[t] { bytes.append(contentsOf: b) } }
        return String(decoding: bytes, as: UTF8.self)
    }
}
