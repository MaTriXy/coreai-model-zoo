import Foundation

/// Accumulates bytes and emits complete lines. A FileHandle's readabilityHandler is invoked
/// serially per handle, so single-threaded access to `buf` is safe (hence @unchecked Sendable).
private final class LineBuffer: @unchecked Sendable {
    private var buf = ""
    private let onLine: @Sendable (String) -> Void
    init(onLine: @escaping @Sendable (String) -> Void) { self.onLine = onLine }
    func feed(_ s: String) {
        buf += s
        while let nl = buf.firstIndex(of: "\n") {
            let line = String(buf[..<nl]); buf.removeSubrange(...nl)
            onLine(line)
        }
    }
}

private final class TextBox: @unchecked Sendable {
    private(set) var text = ""
    func append(_ s: String) { text += s }
    func tail(_ n: Int) -> String { String(text.suffix(n)) }
}

/// Runs the Python Core AI backend (app_backend.py) as a subprocess and surfaces its
/// "PROGRESS <stage> <i> <n>" / "DONE <ply> <splat>" lines as @Published state.
@MainActor
final class Generator: ObservableObject {
    @Published var isRunning = false
    @Published var status = "idle"
    @Published var fraction = 0.0
    @Published var resultPLY: URL?
    @Published var resultSplat: URL?
    @Published var error: String?

    private var proc: Process?

    func run(input: URL, steps: Int, python: String, backendDir: String) {
        isRunning = true; status = "starting…"; fraction = 0
        error = nil; resultPLY = nil; resultSplat = nil

        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposplat_out", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let stamp = Int(Date().timeIntervalSince1970)
        let ply = outDir.appendingPathComponent("gen_\(stamp).ply")
        let splat = outDir.appendingPathComponent("gen_\(stamp).splat")

        let pyPath = (python as NSString).expandingTildeInPath
        let dirPath = (backendDir as NSString).expandingTildeInPath
        let script = (dirPath as NSString).appendingPathComponent("app_backend.py")

        let p = Process()
        p.executableURL = URL(fileURLWithPath: pyPath)
        p.currentDirectoryURL = URL(fileURLWithPath: dirPath)
        p.arguments = [script,
                       "--input", input.path,
                       "--out-ply", ply.path,
                       "--out-splat", splat.path,
                       "--steps", String(steps)]

        let outPipe = Pipe(); let errPipe = Pipe()
        p.standardOutput = outPipe; p.standardError = errPipe

        let stdoutLines = LineBuffer { [weak self] line in
            Task { @MainActor in self?.handle(line, ply: ply, splat: splat) }
        }
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = String(decoding: handle.availableData, as: UTF8.self)
            if !chunk.isEmpty { stdoutLines.feed(chunk) }
        }
        let errBox = TextBox()
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = String(decoding: handle.availableData, as: UTF8.self)
            if !chunk.isEmpty { errBox.append(chunk) }
        }
        p.terminationHandler = { [weak self] proc in
            let code = proc.terminationStatus
            let errTail = errBox.tail(1200)
            Task { @MainActor in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                guard let self else { return }
                self.isRunning = false
                if self.resultPLY == nil {
                    self.error = "backend exited (\(code))\n\(errTail)"
                    self.status = "failed"
                }
            }
        }

        self.proc = p
        do {
            try p.run()
        } catch {
            self.isRunning = false
            self.error = "launch failed: \(error.localizedDescription)\npython: \(pyPath)\nscript: \(script)"
            self.status = "failed"
        }
    }

    func cancel() { proc?.terminate() }

    private func handle(_ line: String, ply: URL, splat: URL) {
        let parts = line.split(separator: " ").map(String.init)
        guard let tag = parts.first else { return }
        if tag == "PROGRESS", parts.count >= 4 {
            let stage = parts[1]
            let i = Double(parts[2]) ?? 0, n = Double(parts[3]) ?? 1
            status = "\(stage) \(parts[2])/\(parts[3])"
            switch stage {
            case "load":       fraction = 0.03
            case "preprocess": fraction = 0.10
            case "encode":     fraction = 0.15
            case "sample":     fraction = 0.20 + 0.70 * (i / max(n, 1))
            case "decode":     fraction = 0.92
            default:           break
            }
        } else if tag == "DONE" {
            status = "done"; fraction = 1.0
            resultPLY = ply
            if FileManager.default.fileExists(atPath: splat.path) { resultSplat = splat }
        }
    }
}
