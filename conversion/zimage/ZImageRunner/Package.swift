// swift-tools-version: 6.0
import PackageDescription

// Host runner for the Z-Image-Turbo port — the SAME ZImagePipeline.swift the macOS app
// uses, driven from a CLI so the Swift host loop can be gated against the Python
// reference engine (conversion/zimage/pipeline_engine.py) without touching a GUI.
//
//   DEVELOPER_DIR=~/Downloads/Xcode-beta.app/Contents/Developer \
//   swift run -c release ZImageRunner --bundle <bundle-dir> --side 512 \
//       --prompt "a red apple on a wooden table, studio lighting" --out /tmp/z.png
let package = Package(
    name: "ZImageRunner",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../../../../coreai-models"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
    ],
    targets: [
        .executableTarget(
            name: "ZImageRunner",
            dependencies: [
                .product(name: "CoreAIShared", package: "coreai-models"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/ZImageRunner"
        )
    ]
)
