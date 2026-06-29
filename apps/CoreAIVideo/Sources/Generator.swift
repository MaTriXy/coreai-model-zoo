import Foundation

/// Accumulates bytes and emits complete lines (FileHandle readabilityHandler is serial per handle).
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

enum GenPhase { case idle, loading, ready, generating, failed }

/// Runs app_backend.py as a RESIDENT process: it loads the Core AI bundles once (prints READY),
/// then we feed it one "<seed>\t<prompt>" line per generation and stream its
/// "PROGRESS <stage> <i> <n>" / "DONE <mp4>" / "ERROR <msg>" output back as @Published state.
@MainActor
final class Generator: ObservableObject {
    @Published var phase: GenPhase = .idle
    @Published var status = "not started"
    @Published var fraction = 0.0
    @Published var resultURL: URL?
    @Published var error: String?

    private var proc: Process?
    private var stdin: FileHandle?

    /// Launch the backend and keep it resident. Call once (idempotent).
    func start(python: String, backendDir: String, runtime: String, coreai: String) {
        guard proc == nil else { return }
        phase = .loading; status = "loading model…"; fraction = 0; error = nil

        let pyPath = (python as NSString).expandingTildeInPath
        let dirPath = (backendDir as NSString).expandingTildeInPath
        let script = (dirPath as NSString).appendingPathComponent("app_backend.py")

        let p = Process()
        p.executableURL = URL(fileURLWithPath: pyPath)
        p.currentDirectoryURL = URL(fileURLWithPath: dirPath)
        p.arguments = [script,
                       "--runtime", (runtime as NSString).expandingTildeInPath,
                       "--coreai", (coreai as NSString).expandingTildeInPath]

        let inPipe = Pipe(); let outPipe = Pipe(); let errPipe = Pipe()
        p.standardInput = inPipe; p.standardOutput = outPipe; p.standardError = errPipe
        self.stdin = inPipe.fileHandleForWriting

        let lines = LineBuffer { [weak self] line in
            Task { @MainActor in self?.handle(line) }
        }
        outPipe.fileHandleForReading.readabilityHandler = { h in
            let s = String(decoding: h.availableData, as: UTF8.self)
            if !s.isEmpty { lines.feed(s) }
        }
        let errBox = TextBox()
        errPipe.fileHandleForReading.readabilityHandler = { h in
            let s = String(decoding: h.availableData, as: UTF8.self)
            if !s.isEmpty { errBox.append(s) }
        }
        p.terminationHandler = { [weak self] proc in
            let code = proc.terminationStatus
            let tail = errBox.tail(1500)
            Task { @MainActor in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                guard let self else { return }
                self.proc = nil; self.stdin = nil
                if self.phase != .ready {  // died before/while a job — surface it
                    self.phase = .failed
                    self.status = "backend exited (\(code))"
                    self.error = "backend exited (\(code))\n\(tail)"
                }
            }
        }

        self.proc = p
        do { try p.run() } catch {
            phase = .failed; status = "launch failed"
            self.error = "launch failed: \(error.localizedDescription)\npython: \(pyPath)\nscript: \(script)"
        }
    }

    /// Submit one generation. Requires phase == .ready.
    func generate(prompt: String, seed: Int) {
        guard phase == .ready, let stdin else { return }
        resultURL = nil; error = nil
        phase = .generating; status = "starting…"; fraction = 0.02
        let line = "\(seed)\t\(prompt.replacingOccurrences(of: "\n", with: " "))\n"
        stdin.write(Data(line.utf8))
    }

    func stop() { proc?.terminate() }

    private func handle(_ line: String) {
        let parts = line.split(separator: " ").map(String.init)
        guard let tag = parts.first else { return }
        switch tag {
        case "READY":
            phase = .ready; status = "ready"; fraction = 0
        case "PROGRESS" where parts.count >= 4:
            let stage = parts[1]
            let i = Double(parts[2]) ?? 0, n = Double(parts[3]) ?? 1
            status = stage == "sample" ? "sampling \(parts[2])/\(parts[3])" : stage
            switch stage {
            case "load":   fraction = 0.02 + 0.08 * (i / max(n, 1)); phase = .loading
            case "encode": fraction = 0.06
            case "sample": fraction = 0.10 + 0.80 * (i / max(n, 1))
            case "decode": fraction = 0.92
            default: break
            }
        case "DONE" where parts.count >= 2:
            let path = parts.dropFirst().joined(separator: " ")
            resultURL = URL(fileURLWithPath: path)
            phase = .ready; status = "done"; fraction = 1.0
        case "ERROR":
            error = parts.dropFirst().joined(separator: " ")
            phase = .ready; status = "error"; fraction = 0
        default:
            break
        }
    }
}
