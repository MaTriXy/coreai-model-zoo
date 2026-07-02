// CameraFeed — capture session + a 16-frame ring sampled at ~10 fps (≈1.6 s window), each frame
// stored as a small CGImage. Retaining raw CVPixelBuffers would stall the capture pool, so frames
// are converted immediately on the delegate queue.
import AVFoundation
import CoreImage
import SwiftUI

final class CameraFeed: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.feed")
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let lock = NSLock()
    private var ring: [CGImage] = []
    private var lastStore = ContinuousClock.now
    private let interval = Duration.milliseconds(100)

    func start() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            guard await AVCaptureDevice.requestAccess(for: .video) else { return false }
        } else if status != .authorized {
            return false
        }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(for: .video),
            let input = try? AVCaptureDeviceInput(device: device) else { return false }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        if session.canAddInput(input) { session.addInput(input) }
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
            #if os(iOS)
            conn.videoRotationAngle = 90                       // portrait
            #endif
        }
        session.commitConfiguration()
        let s = session
        return await Task.detached { s.startRunning(); return s.isRunning }.value
    }

    func stop() {
        let s = session
        Task.detached { s.stopRunning() }
        lock.lock(); ring.removeAll(); lock.unlock()
    }

    /// Latest 16 frames (oldest→newest), or nil until the ring fills.
    func snapshot() -> [CGImage]? {
        lock.lock(); defer { lock.unlock() }
        return ring.count >= ActionEngine.frames ? Array(ring.suffix(ActionEngine.frames)) : nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = ContinuousClock.now
        guard now - lastStore >= interval, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastStore = now
        // downscale so the shorter side is ~256 before storing (keeps the ring tiny)
        let ci = CIImage(cvPixelBuffer: pb)
        let scale = 256.0 / min(ci.extent.width, ci.extent.height)
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cg = ciContext.createCGImage(scaled, from: scaled.extent) else { return }
        lock.lock()
        ring.append(cg)
        if ring.count > ActionEngine.frames { ring.removeFirst(ring.count - ActionEngine.frames) }
        lock.unlock()
    }
}

// SwiftUI camera preview (both platforms).
#if os(iOS)
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
    final class PreviewView: UIView {
        override static var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
#else
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        v.wantsLayer = true
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = v.bounds
        layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        v.layer?.addSublayer(layer)
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
#endif
