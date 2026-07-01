import Foundation
import SceneKit

/// Minimal binary-little-endian 3DGS .ply loader -> colored SceneKit point cloud.
/// Reads x/y/z + f_dc_0..2 (SH DC -> RGB) + opacity (sigmoid alpha for culling).
enum PLY {
    struct Cloud { var xyz: [Float]; var rgb: [Float]; var count: Int }
    enum PLYError: Error { case badHeader }

    static func load(_ url: URL, alphaThreshold: Float = 0.04, maxPoints: Int = 400_000) throws -> Cloud {
        let data = try Data(contentsOf: url)
        guard let hdr = data.range(of: Data("end_header\n".utf8)) else { throw PLYError.badHeader }
        let header = String(decoding: data[..<hdr.upperBound], as: UTF8.self)

        var n = 0
        var props: [String] = []
        for raw in header.split(separator: "\n") {
            let line = raw.split(separator: " ")
            if raw.hasPrefix("element vertex"), line.count >= 3 { n = Int(line[2]) ?? 0 }
            else if raw.hasPrefix("property"), let name = line.last { props.append(String(name)) }
        }
        guard n > 0,
              let ix = props.firstIndex(of: "x"), let iy = props.firstIndex(of: "y"), let iz = props.firstIndex(of: "z"),
              let ir = props.firstIndex(of: "f_dc_0"), let ig = props.firstIndex(of: "f_dc_1"), let ib = props.firstIndex(of: "f_dc_2"),
              let io = props.firstIndex(of: "opacity")
        else { throw PLYError.badHeader }

        let stride = props.count
        let floats: [Float32] = data[hdr.upperBound...].withUnsafeBytes {
            Array($0.bindMemory(to: Float32.self).prefix(n * stride))
        }

        let C0: Float = 0.282_094_79
        let step = max(1, n / maxPoints)
        var xyz: [Float] = []; var rgb: [Float] = []
        xyz.reserveCapacity(n / step * 3); rgb.reserveCapacity(n / step * 3)

        var sx: Float = 0, sy: Float = 0, sz: Float = 0
        var kept = 0
        var i = 0
        while i < n {
            let b = i * stride
            let alpha = 1 / (1 + expf(-floats[b + io]))
            if alpha > alphaThreshold {
                let x = floats[b + ix], y = floats[b + iy], z = floats[b + iz]
                xyz.append(x); xyz.append(y); xyz.append(z)
                sx += x; sy += y; sz += z; kept += 1
                rgb.append(min(max(0.5 + C0 * floats[b + ir], 0), 1))
                rgb.append(min(max(0.5 + C0 * floats[b + ig], 0), 1))
                rgb.append(min(max(0.5 + C0 * floats[b + ib], 0), 1))
            }
            i += step
        }
        // Center on centroid so camera orbit feels right.
        if kept > 0 {
            let cx = sx / Float(kept), cy = sy / Float(kept), cz = sz / Float(kept)
            var j = 0
            while j < xyz.count { xyz[j] -= cx; xyz[j + 1] -= cy; xyz[j + 2] -= cz; j += 3 }
        }
        return Cloud(xyz: xyz, rgb: rgb, count: kept)
    }

    static func scene(from cloud: Cloud) -> SCNScene {
        let posData = cloud.xyz.withUnsafeBytes { Data($0) }
        let colData = cloud.rgb.withUnsafeBytes { Data($0) }
        let count = cloud.count

        let vSource = SCNGeometrySource(
            data: posData, semantic: .vertex, vectorCount: count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let cSource = SCNGeometrySource(
            data: colData, semantic: .color, vectorCount: count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12)

        let indices = (0..<UInt32(count)).map { $0 }
        let idxData = indices.withUnsafeBytes { Data($0) }
        let element = SCNGeometryElement(
            data: idxData, primitiveType: .point,
            primitiveCount: count, bytesPerIndex: 4)
        element.pointSize = 6
        element.minimumPointScreenSpaceRadius = 1.5
        element.maximumPointScreenSpaceRadius = 4

        let geo = SCNGeometry(sources: [vSource, cSource], elements: [element])
        geo.firstMaterial?.lightingModel = .constant
        geo.firstMaterial?.isLitPerPixel = false

        let node = SCNNode(geometry: geo)
        node.scale = SCNVector3(1, -1, 1)   // TripoSplat space is y-down

        let scene = SCNScene()
        scene.rootNode.addChildNode(node)
        scene.background.contents = NSColor.black
        return scene
    }
}
