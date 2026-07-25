// ModelDownloader.swift — in-app model delivery for the Core AI sample apps.
//
// Downloads `.aimodel` bundles (directory trees) from a Hugging Face model repo into a
// destination directory. Every file streams into a hidden staging directory and a bundle is
// renamed into place only when ALL of its files are complete: AIModel must never see a partial
// bundle — a partially-present bundle poisons the content-keyed coreai-cache (later loads fail
// ENOENT until the cache is wiped).
//
// The payload of these bundles is one huge `main.mlirb` (e.g. ~30 GB for the Mac models) plus a
// few tiny tokenizer files, so the delivery is built around RANGE-CHUNKED PARALLELISM with
// CROSS-LAUNCH RESUME:
//   • Parallel: every file >`chunkSize` is split into byte-range chunks pulled CONCURRENTLY (up to
//     `maxConnections`) — a single HF/CDN connection is bandwidth-capped, so 6 connections on the
//     one big file is several times faster than a one-stream-per-file loop (which gave the 30 GB
//     file zero parallelism). Each chunk is written straight into its final offset of the staging
//     file (a FileHandle seek+write), so there is no concatenation pass and disk use stays at 1×
//     (peak RAM = chunkSize × maxConnections).
//   • Resumable across launches: each file has a sidecar bitmap (one byte per chunk) under
//     `<dest>/.dl-progress/<local>/…`. A chunk marks its bit done only AFTER its data is written
//     and closed, so `bit==1 ⟹ bytes present`. A re-`fetch` of an interrupted bundle keeps the
//     staging tree, reads the bitmaps, and re-downloads only the missing chunks — so a quit/crash
//     (or the OS killing the app) costs at most the one in-flight chunk, never the whole 30 GB.
//   • Background-tolerant (iOS): the whole fetch holds a UIKit background-task assertion, so a
//     brief trip to the background (app switcher, screen lock) doesn't drop the transfer. A longer
//     suspension that gets the app killed is covered by the resume above on next launch.
//
// Usage:
//   let items = [ModelDownloader.Item(remote: "ios-gpu/foo.aimodel", local: "foo.aimodel")]
//   await downloader.fetch(repo: "https://huggingface.co/org/repo", items: items, into: docs)
//
// `remote` is a directory (or file) path inside the repo; `local` is the name created under the
// destination. Items already present at the destination are skipped (they were placed
// atomically, so presence == complete). Downloaded bundles are excluded from iCloud backup.

import Foundation
#if canImport(UIKit)
import UIKit
#endif

@MainActor
final class ModelDownloader: ObservableObject {
    struct Item {
        let remote: String   // path inside the HF repo
        let local: String    // name under the destination directory
    }

    enum Phase: Equatable {
        case idle, listing, downloading(file: String), failed(String), done
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var fraction: Double = 0
    @Published private(set) var detail = ""

    var busy: Bool {
        switch phase { case .listing, .downloading: return true; default: return false }
    }

    // Size of the session pool = number of independent TCP connections streaming chunks at once.
    // Each is a separate URLSession (one HTTP/2 connection each), so aggregate throughput scales with
    // this until the device radio / bandwidth-delay-product caps it; on-device 12 measured > 6. 8 is
    // a safe default (well under HF's request-COUNT rate limit, the only per-client cap).
    private nonisolated static let maxConnections = 8
    // Files larger than this are split into byte-range chunks of this size. 16 MiB keeps peak RAM
    // (chunkSize × maxConnections ≈ 96 MB) modest while making the progress bar advance smoothly.
    private nonisolated static let chunkSize: Int64 = 16 * 1024 * 1024
    // Per-chunk retry budget (a failed chunk re-fetches only its own ≤chunkSize slice). Each retry
    // re-hits the HF resolve URL, so it re-rolls onto a possibly-different (live) CDN.
    private nonisolated static let maxChunkRetries = 6
    // Per-chunk wall-clock deadline (passed to `dataWithDeadline`). It must absorb a COLD Xet read:
    // HF now stores these LFS files in Xet content-addressed storage, and the FIRST access to a byte
    // range of a freshly-uploaded large file is served by reconstructing it on the fly, which can
    // crawl (measured ~0.8 MB/s cold for some ranges of the 2 GB Gemma-⚡ bundle vs ~4.5 MB/s once the
    // edge is warm). A flat 30 s for a 16 MiB chunk was a 0.5 MB/s floor — the cold ranges tripped it
    // on a mobile link and failed the whole bundle ("chunk stalled > 30s"). Express it as a low
    // throughput floor instead so it tracks `chunkSize`: ~0.18 MB/s ⇒ ~91 s for 16 MiB. Patience is
    // safe — a truly wedged (0-byte) transfer is still cut fast by the 25 s idle timeout below, and a
    // cut chunk's retry usually lands quickly because the first touch warmed the edge cache.
    private nonisolated static let chunkDeadlineSeconds: UInt64 = UInt64(chunkSize / (180 * 1024))
    // Re-applies the Range header across the HF→CDN 302. Without it a redirect that dropped Range
    // would answer 200 with the WHOLE file, and data(for:) would buffer ~30 GB into RAM and crash.
    private nonisolated static let redirector = RangePreservingRedirector()
    // Hidden directory under `dest` holding the per-file completed-chunk bitmaps.
    private nonisolated static let progressDirName = ".dl-progress"
    // Global single-flight guard: two overlapping fetches of the same files open duplicate
    // connections to the HF CDN that fight over the same byte ranges and knock each other out with
    // -1005 "network connection lost" — stalling progress at ~0. Only one fetch runs at a time.
    @MainActor private static var fetchInProgress = false

    private var totalBytes: Int64 = 0
    private var completedBytes: Int64 = 0
    private var shownFraction = -1.0

    // One byte-range of one file: where to GET it, where to write it, and which bitmap bit marks it
    // done. Files at or below `chunkSize` are a single un-ranged segment (a plain whole-file GET).
    private struct Segment: Sendable {
        let fileIndex: Int      // which file (for per-file completion accounting)
        let bundleIndex: Int    // which bundle (for the atomic rename)
        let url: URL
        let dest: URL           // staging file this chunk writes into
        let offset: Int64       // byte offset within the file / dest
        let length: Int64
        let ranged: Bool        // send a Range header (false = whole small file)
        let chunkIndex: Int     // index into the file's bitmap
        let bits: URL           // sidecar bitmap file (one byte per chunk)
    }

    // Accepts "https://huggingface.co/<org>/<name>[/...]" or a bare "<org>/<name>".
    nonisolated static func repoId(from s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let u = URL(string: t), let host = u.host, host.hasSuffix("huggingface.co") {
            let parts = u.path.split(separator: "/").map(String.init)
            return parts.count >= 2 ? "\(parts[0])/\(parts[1])" : nil
        }
        let parts = t.split(separator: "/").map(String.init)
        return parts.count == 2 ? t : nil
    }

    func fetch(repo: String, items: [Item], into dest: URL) async {
        guard let repoId = Self.repoId(from: repo) else {
            phase = .failed("not a Hugging Face repo URL"); return
        }
        let fm = FileManager.default
        let pending = items.filter { !fm.fileExists(atPath: dest.appendingPathComponent($0.local).path) }
        guard !pending.isEmpty else { phase = .done; return }
        // Single-flight: ignore a re-entrant fetch while one is already running (the in-flight one
        // keeps going and resumes any leftover on the next launch).
        if Self.fetchInProgress { return }
        Self.fetchInProgress = true
        defer { Self.fetchInProgress = false }
        phase = .listing; fraction = 0; shownFraction = -1; completedBytes = 0; totalBytes = 0
        detail = "listing files…"

        // Hold a background-task assertion for the whole fetch so a brief trip to the background
        // (app switcher / screen lock) doesn't get the app suspended out from under the download.
        let bg = BackgroundAssertion(name: "model-download")
        defer { bg.end() }

        let cfg = URLSessionConfiguration.default
        cfg.httpMaximumConnectionsPerHost = 1
        // waitsForConnectivity MUST be false. On iOS a dead CDN connection drops with -1005 "network
        // connection lost"; with waitsForConnectivity=true the retry then blocks forever "waiting for
        // connectivity" (the request timeout doesn't run while waiting) → the download hangs silently
        // at 0 bytes. False makes a dead connection fail fast so the retry re-opens a fresh one.
        cfg.waitsForConnectivity = false
        // IDLE timeout (resets on each received packet); the per-chunk wall-clock deadline below is
        // what actually caps a stalled/crawling transfer.
        cfg.timeoutIntervalForRequest = 25
        cfg.timeoutIntervalForResource = 7 * 24 * 60 * 60   // a big set over a slow link can take a while

        // A POOL of independent sessions = independent TCP connections. A single URLSession
        // multiplexes ALL its tasks onto ONE HTTP/2 connection per host (one congestion + flow-control
        // window ≈ one stream's throughput; `httpMaximumConnectionsPerHost` is ignored under H2), so
        // fanning N chunks over one session caps at ~1 stream and collapses (-1005) when overloaded.
        // Giving each fan-out slot its OWN long-lived session makes aggregate throughput scale with N
        // (AWS S3: "spread requests over separate connections… higher aggregate throughput").
        let pool = (0..<max(1, Self.maxConnections)).map { _ in URLSession(configuration: cfg) }
        defer { pool.forEach { $0.invalidateAndCancel() } }

        do {
            // Plan: list each bundle's files, then for every file load its completed-chunk bitmap
            // (resuming a prior partial) or start it fresh, and enqueue only the missing chunks.
            var plan: [(item: Item, files: [PlannedFile])] = []
            for item in pending { plan.append((item, try await Self.list(repoId: repoId, item: item))) }

            let progressRoot = dest.appendingPathComponent(Self.progressDirName)
            var bundleStaging: [URL] = []
            var bundleFinal: [URL] = []
            var bundleProgress: [URL] = []
            var bundleRemaining: [Int] = []     // files still missing chunks, per bundle
            var fileRemaining: [Int] = []       // chunks still missing, per global fileIndex
            var segments: [Segment] = []
            var fileIndex = 0
            for (bi, entry) in plan.enumerated() {
                // Keep an existing staging tree so a re-fetch can resume into it.
                let st = dest.appendingPathComponent(".staging-\(entry.item.local)")
                let prog = progressRoot.appendingPathComponent(entry.item.local)
                try fm.createDirectory(at: st, withIntermediateDirectories: true)
                try fm.createDirectory(at: prog, withIntermediateDirectories: true)
                Self.excludeFromBackup(st); Self.excludeFromBackup(progressRoot)
                bundleStaging.append(st)
                bundleFinal.append(dest.appendingPathComponent(entry.item.local))
                bundleProgress.append(prog)

                var bundlePendingFiles = 0
                for f in entry.files {
                    let destFile = st.appendingPathComponent(f.rel)
                    let bits = prog.appendingPathComponent(f.rel + ".bits")
                    try fm.createDirectory(at: destFile.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try fm.createDirectory(at: bits.deletingLastPathComponent(), withIntermediateDirectories: true)

                    let geo = Self.chunkGeometry(f.size)
                    // Trust a prior partial only if its bitmap matches this file's chunking AND the
                    // staging file is still there; otherwise reset this file (empty file + zero map).
                    var done = [UInt8](repeating: 0, count: geo.count)
                    if let saved = try? Data(contentsOf: bits), saved.count == geo.count,
                       fm.fileExists(atPath: destFile.path) {
                        done = [UInt8](saved)
                    } else {
                        fm.createFile(atPath: destFile.path, contents: nil)
                        try Data(count: geo.count).write(to: bits)
                    }

                    var filePending = 0
                    for (ci, g) in geo.enumerated() {
                        totalBytes += g.length
                        if done[ci] != 0 {
                            completedBytes += g.length      // already on disk from a prior run
                        } else {
                            segments.append(Segment(fileIndex: fileIndex, bundleIndex: bi, url: f.url,
                                                    dest: destFile, offset: g.offset, length: g.length,
                                                    ranged: g.ranged, chunkIndex: ci, bits: bits))
                            filePending += 1
                        }
                    }
                    fileRemaining.append(filePending)
                    if filePending > 0 { bundlePendingFiles += 1 }
                    fileIndex += 1
                }
                bundleRemaining.append(bundlePendingFiles)
            }

            // Place any bundle that needs nothing — empty, or fully present from a prior run that
            // was interrupted right before the rename.
            for bi in plan.indices where bundleRemaining[bi] == 0 {
                try Self.placeBundle(bundleStaging[bi], at: bundleFinal[bi], progress: bundleProgress[bi])
            }
            recomputeProgress()

            phase = .downloading(file: "")
            // Bounded fan-out: keep `maxConnections` chunks in flight, refilling as each lands.
            // Bookkeeping runs serially here on the main actor; a thrown chunk cancels the rest
            // (data(for:) is cancellation-aware) and falls through to the catch.
            // Each fan-out slot is pinned to its own pool session (= its own connection); when a
            // slot's chunk lands, the next chunk reuses that same session so the connection persists.
            try await withThrowingTaskGroup(of: (Segment, Int).self) { group in
                var iterator = segments.makeIterator()
                var inFlight = 0
                var slot = 0
                for _ in 0..<pool.count {
                    guard let seg = iterator.next() else { break }
                    let s = slot; slot += 1
                    let sess = pool[s]
                    group.addTask { try await Self.fetchChunk(seg, via: sess, deadline: Self.chunkDeadlineSeconds); return (seg, s) }
                    inFlight += 1
                }
                while inFlight > 0 {
                    let (seg, freed) = try await group.next()!
                    inFlight -= 1
                    completedBytes += seg.length
                    recomputeProgress()
                    fileRemaining[seg.fileIndex] -= 1
                    if fileRemaining[seg.fileIndex] == 0 {     // file fully written
                        bundleRemaining[seg.bundleIndex] -= 1
                        if bundleRemaining[seg.bundleIndex] == 0 {   // bundle complete -> place it
                            try Self.placeBundle(bundleStaging[seg.bundleIndex], at: bundleFinal[seg.bundleIndex],
                                                 progress: bundleProgress[seg.bundleIndex])
                        }
                    }
                    if let next = iterator.next() {
                        let sess = pool[freed]
                        group.addTask { try await Self.fetchChunk(next, via: sess, deadline: Self.chunkDeadlineSeconds); return (next, freed) }
                        inFlight += 1
                    }
                }
            }

            try? fm.removeItem(at: progressRoot)    // best-effort sweep of the now-empty bitmap root
            fraction = 1; detail = ""
            phase = .done
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func recomputeProgress() {
        let f = totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 0
        guard f - shownFraction >= 0.002 || f >= 1 else { return }   // ~50 UI updates per download
        shownFraction = f; fraction = min(f, 1)
        detail = "\(Self.fmt(completedBytes)) / \(Self.fmt(totalBytes))"
    }

    // Atomic placement: rename the clean staging tree into its final name, drop the bitmap dir.
    private nonisolated static func placeBundle(_ staging: URL, at final: URL, progress: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: final)
        try fm.moveItem(at: staging, to: final)      // surface a real placement failure as .failed
        try? fm.removeItem(at: progress)
        excludeFromBackup(final)
    }

    private nonisolated static func excludeFromBackup(_ url: URL) {
        var v = URLResourceValues(); v.isExcludedFromBackup = true
        var u = url; try? u.setResourceValues(v)
    }

    // Byte-range geometry for a file (or one whole-file segment if small enough).
    private nonisolated static func chunkGeometry(_ size: Int64) -> [(offset: Int64, length: Int64, ranged: Bool)] {
        guard size > chunkSize else { return [(0, max(size, 0), false)] }
        var out: [(Int64, Int64, Bool)] = []
        var off: Int64 = 0
        while off < size {
            let len = min(chunkSize, size - off)
            out.append((off, len, true))
            off += len
        }
        return out
    }

    // Fetch one chunk, write it at its offset in the staging file, then mark its bitmap bit. The
    // data write+close happens BEFORE the bit is set so a crash can never leave bit==1 over missing
    // bytes (a re-download just rewrites the same slice). Retries re-fetch only this chunk.
    private nonisolated static func fetchChunk(_ seg: Segment, via session: URLSession, deadline: UInt64) async throws {
        var attempt = 0
        while true {
            do {
                var req = URLRequest(url: seg.url)
                if seg.ranged {
                    req.setValue("bytes=\(seg.offset)-\(seg.offset + seg.length - 1)",
                                 forHTTPHeaderField: "Range")
                }
                let (data, resp) = try await Self.dataWithDeadline(req, via: session, seconds: deadline)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
                // A ranged GET must answer 206; a whole-file GET 200. Anything else (e.g. the CDN
                // ignored the Range and sent 200) would mis-place bytes, so fail loudly instead.
                guard seg.ranged ? code == 206 : code == 200 else {
                    throw err("HTTP \(code) for \(seg.url.lastPathComponent)")
                }
                let fh = try FileHandle(forWritingTo: seg.dest)
                do {
                    try fh.seek(toOffset: UInt64(seg.offset))
                    try fh.write(contentsOf: data)
                    try fh.close()
                } catch { try? fh.close(); throw error }
                // Bytes are durable in the OS file now → record the chunk as done.
                let bh = try FileHandle(forWritingTo: seg.bits)
                do {
                    try bh.seek(toOffset: UInt64(seg.chunkIndex))
                    try bh.write(contentsOf: Data([1]))
                    try bh.close()
                } catch { try? bh.close(); throw error }
                return
            } catch {
                if Task.isCancelled { throw error }     // group is tearing down — don't retry
                attempt += 1
                if attempt > maxChunkRetries { throw error }
                try? await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)  // 0.5s, 1s, …
            }
        }
    }

    // A wall-clock deadline around one GET (redirect + body). `timeoutIntervalForRequest` is only an
    // IDLE timer that resets on every received byte, so a connection that degrades to a crawl (HF's
    // shared HTTP/2 connection sometimes does this, stalling all in-flight chunks at once) never
    // trips it and the download wedges at a fixed byte count. This races the transfer against a hard
    // timeout and cancels it, so a stalled/crawling chunk fails fast and the retry re-rolls the HF
    // resolve onto a fresh connection (and possibly a different CDN). The deadline (`chunkDeadlineSeconds`,
    // ~91 s for 16 MiB) is deliberately patient so a cold Xet reconstruction isn't mistaken for a stall;
    // a genuine 0-byte wedge is cut far sooner by the 25 s idle `timeoutIntervalForRequest`.
    private nonisolated static func dataWithDeadline(_ req: URLRequest, via session: URLSession,
                                                     seconds: UInt64) async throws -> (Data, URLResponse) {
        try await withThrowingTaskGroup(of: (Data, URLResponse).self) { group in
            group.addTask { try await session.data(for: req, delegate: redirector) }
            group.addTask {
                try await Task.sleep(nanoseconds: seconds * 1_000_000_000)
                throw err("chunk stalled > \(seconds)s")
            }
            defer { group.cancelAll() }                 // cancel the loser (transfer or timer)
            guard let first = try await group.next() else { throw err("no chunk result") }
            return first
        }
    }

    // MARK: repo listing

    private struct PlannedFile: Sendable { let url: URL; let rel: String; let size: Int64 }
    private struct TreeEntry: Decodable {
        let type: String, path: String, size: Int64?
        let lfs: LFS?
        struct LFS: Decodable { let size: Int64? }
    }

    // Enumerate the files under `item.remote` via the HF tree API (paths here hold a handful of
    // files — pagination is not handled).
    private nonisolated static func list(repoId: String, item: Item) async throws -> [PlannedFile] {
        guard let api = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main/\(item.remote)?recursive=true") else {
            throw err("bad path \(item.remote)")
        }
        let (data, resp) = try await URLSession.shared.data(from: api)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw err("\(item.remote): not found in \(repoId)")
        }
        let entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        let prefix = item.remote.hasSuffix("/") ? item.remote : item.remote + "/"
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = e.path == item.remote ? (e.path as NSString).lastPathComponent
                                            : String(e.path.dropFirst(prefix.count))
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(e.path)") else {
                throw err("bad file path \(e.path)")
            }
            return PlannedFile(url: url, rel: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }

    private nonisolated static func err(_ msg: String) -> Error {
        NSError(domain: "ModelDownloader", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private nonisolated static func fmt(_ b: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: b, countStyle: .file)
    }
}

// A no-op-on-macOS wrapper around a UIKit finite-length background-task assertion. Holding one
// keeps the app from being suspended for a short grace period after it leaves the foreground, so a
// quick app-switch mid-download doesn't drop the connection. (macOS apps aren't suspended this way,
// so there's nothing to hold there.)
private struct BackgroundAssertion {
    #if canImport(UIKit)
    private let id: UIBackgroundTaskIdentifier
    @MainActor init(name: String) {
        var handle: UIBackgroundTaskIdentifier = .invalid
        handle = UIApplication.shared.beginBackgroundTask(withName: name) {
            if handle != .invalid { UIApplication.shared.endBackgroundTask(handle) }
        }
        id = handle
    }
    @MainActor func end() {
        if id != .invalid { UIApplication.shared.endBackgroundTask(id) }
    }
    #else
    init(name: String) {}
    func end() {}
    #endif
}

// Carries the `Range` header onto the redirected request. HF `resolve/...` 302-redirects to the
// CDN; URLSession usually copies headers across a redirect, but if it ever dropped Range the CDN
// would send the full file (200) and the chunk's data(for:) would buffer the whole ~30 GB. This
// guarantees the redirected GET stays a 206 partial.
private final class RangePreservingRedirector: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        var req = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            req.setValue(range, forHTTPHeaderField: "Range")
        }
        completionHandler(req)
    }
}
