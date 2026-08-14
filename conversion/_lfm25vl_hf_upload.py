"""Stage + upload the LFM2.5-VL Core AI bundles to HF. USER-GATED.

Run: `python _lfm25vl_hf_upload.py [450m|3b] [--dry-run]`.

Publishes three bundles: the fp16 vision tower, the int8lin VLM decoder (the pair
that IS the model), and the text core -- the same decoder weights with no image
input, which is both a usable 350M LFM2 text bundle and the only configuration
`llm-benchmark` / `llm-runner` can drive, since neither can bind an image buffer.

Deliberately NOT published: `int4lin` (0/9 gate cases, fluent drift -- see the card)
and the int8 vision tower (smaller but slower AND less accurate on Mac; it may still
be the right call on a phone, which is unmeasured).

Layout matches the rest of the catalog (`conversion/_hf_catalog.py` reads it):

    gpu-pipelined/<bundle>/<bundle>.aimodel/         + metadata.json + tokenizer/
    ios-h18p/<bundle>/<bundle>.h18p.aimodelc/        + metadata.json + tokenizer/
    config.json      -- the source config, so zoo_verify can compare against it
    LICENSE          -- carried from the source repo
    README.md        -- the card

The iOS subtree carries its own `metadata.json` (assets.main points at the
`.aimodelc`, which `AIModel(contentsOf:)` auto-detects as AOT and does not
re-JIT). Build it after the Mac export with:

    xcrun coreai-build compile exports/<name>/<name>.aimodel \
        --platform iOS --preferred-compute gpu --architecture h18p \
        --output exports/<name>_ios

No `--expect-frequent-reshapes` on iOS: it makes the runtime discard the AOT
specialization and compile on device, which SIGSEGVs with no log.
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402

SIZES = {
    "450m": {
        "repo": "mlboydaisuke/LFM2.5-VL-450M-CoreAI",
        "source": "LiquidAI/LFM2.5-VL-450M",
        "card": "HF_README.md",
        # The pair that IS the model, plus the text core (a usable 350M LFM2 text bundle and
        # the only configuration llm-benchmark/llm-runner can drive).
        "bundles": [
            "lfm2_5_vl_450m_vision_fp16",
            "lfm2_5_vl_450m_decode_int8lin",
            "lfm2_5_vl_450m_decode_int8lin_textcore",
        ],
        # Device-gated on an iPhone 17 Pro.
        "ios": [
            "lfm2_5_vl_450m_decode_int8lin",
            "lfm2_5_vl_450m_decode_int8lin_textcore",
            "lfm2_5_vl_450m_vision_fp16",
        ],
        "message": ("LFM2.5-VL-450M -> Core AI: fp16 SigLIP2-NaFlex tower + int8lin "
                    "decoder (658 MB pair) + text core; oracle gate 16/16"),
    },
    "3b": {
        "repo": "mlboydaisuke/LFM2.5-VL-3B-CoreAI",
        "source": "LiquidAI/LFM2.5-VL-3B",
        "card": "HF_README_3B.md",
        # int8lin is the Mac ship; int4lin is published beside it because it costs this
        # model nothing on the suite (7/9, same as fp16) and is the only variant whose AOT
        # fits under the iOS 2 GiB load wall.
        "bundles": [
            "lfm2_5_vl_3b_vision_fp16",
            "lfm2_5_vl_3b_decode_int8lin",
            "lfm2_5_vl_3b_decode_int4lin",
            "lfm2_5_vl_3b_decode_int8lin_textcore",
        ],
        "ios": [],  # nothing until a phone has run it
        "message": ("LFM2.5-VL-3B -> Core AI: fp16 SigLIP2-NaFlex tower + int8lin/int4lin "
                    "decoders + text core; fp32 ladder cos 1.000000, suite 7/9 = fp16 baseline"),
    },
}

def stage(cfg: dict, stage_dir: Path) -> Path:
    STAGE = stage_dir
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "gpu-pipelined").mkdir(parents=True)

    for name in cfg["bundles"]:
        src = exports_dir() / name
        if not src.is_dir():
            raise SystemExit(f"missing bundle {src} -- export it first")
        shutil.copytree(src, STAGE / "gpu-pipelined" / name)
        print(f"  staged {name}")

    # iOS AOT variants, for the bundles that were device-gated. Same directory
    # shape, metadata pointed at the .aimodelc.
    for name in cfg["ios"]:
        aotc = exports_dir() / f"{name}_ios" / f"{name}.h18p.aimodelc"
        if not aotc.is_dir():
            print(f"  SKIP ios-h18p/{name} (no {aotc.name}; run coreai-build compile)")
            continue
        dst = STAGE / "ios-h18p" / name
        dst.mkdir(parents=True)
        shutil.copytree(aotc, dst / aotc.name)
        meta_path = exports_dir() / name / "metadata.json"
        if meta_path.exists():  # the vision tower is a bare .aimodel, no bundle metadata
            meta = json.loads(meta_path.read_text())
            meta["assets"]["main"] = aotc.name
            (dst / "metadata.json").write_text(json.dumps(meta, indent=1))
        tokenizer = exports_dir() / name / "tokenizer"
        if tokenizer.is_dir():
            shutil.copytree(tokenizer, dst / "tokenizer")
        print(f"  staged ios-h18p/{name}")

    snap = Path(hf_snapshot(cfg["source"]))
    shutil.copy2(snap / "config.json", STAGE / "config.json")
    for license_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if (snap / license_name).exists():
            shutil.copy2(snap / license_name, STAGE / "LICENSE")
            break
    else:
        print("  WARNING: no LICENSE in the source snapshot")
    shutil.copy2(Path(__file__).parents[1] / "models" / "lfm2.5-vl" / cfg["card"],
                 STAGE / "README.md")

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({total / 1e9:.2f} GB)")
    return STAGE


def main() -> None:
    size = next((a for a in sys.argv[1:] if a in SIZES), "450m")
    cfg = SIZES[size]
    folder = stage(cfg, Path(os.environ.get("ZOO_STAGE", "/tmp")) / f"lfm25vl_hf_{size}")
    if "--dry-run" in sys.argv:
        print("dry run -- not uploading")
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(folder)}  {p.stat().st_size / 1e6:.1f} MB")
        return

    api = HfApi()
    api.create_repo(cfg["repo"], repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=cfg["repo"], repo_type="model",
        commit_message=cfg["message"])
    print("uploaded", cfg["repo"])
    print(json.dumps({"repo": cfg["repo"], "bundles": cfg["bundles"],
                      "ios": cfg["ios"]}, indent=1))


if __name__ == "__main__":
    main()
