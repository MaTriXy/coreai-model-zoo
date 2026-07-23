# Community port — NOT an Apple model.
"""Export the VibeVoice acoustic-tokenizer DECODER (causal ConvNeXt VAE vocoder) to Core AI .aimodel
and engine-gate vs the non-streaming golden (artifacts/dec_ref.npz).

  latents[1,64,T] -> audio[1,1,3200*T]   (whole-sequence non-streaming; == streaming, cos 1.0)

fp16 (quant-sensitive continuous vocoder, VoxCPM/dots lesson). --tframes selects the fixed export T.

  PYTHONPATH=. <coreai-venv>/bin/python export_decoder.py --tframes 30
"""
from __future__ import annotations
import argparse, asyncio, shutil, subprocess, sys, glob
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from decoder_ref import DecoderOverlay  # noqa: E402

ART = HERE / "artifacts"
SNAP = "/Users/majimadaisuke/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    n = min(a.numel(), b.numel())
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


async def run(tframes, DT):
    import coreai.runtime as rt
    z = np.load(ART / "dec_ref.npz")
    sd = load_file(glob.glob(SNAP + "/*.safetensors")[0])
    lat_all = torch.from_numpy(z["latents_all"])           # (1,64,N)
    N = lat_all.shape[-1]
    T = tframes
    if T <= N:
        lat = lat_all[:, :, :T].to(DT)
    else:
        # the golden holds only N latents; tile them out to the requested fixed export length.
        # The decoder is causal, so the first N output frames are unaffected by what follows —
        # the gate below still compares against the full golden audio.
        lat = lat_all.repeat(1, 1, -(-T // N))[:, :, :T].to(DT)
    gold = z["audio_full"][:, :, :min(T, N) * 3200]

    dec = DecoderOverlay().to(DT).eval().load_upstream(sd)
    with torch.inference_mode():
        t_out = dec(lat).float().numpy()
    print(f"[dec] torch overlay(T={T}) vs golden: cos={cos(t_out, gold):.6f}  "
          f"(fp32 sanity below)")
    dec32 = DecoderOverlay().to(torch.float32).eval().load_upstream(sd)
    with torch.inference_mode():
        c32 = cos(dec32(lat.float()).numpy(), gold)
    print(f"[dec] torch overlay fp32 vs golden: cos={c32:.6f}")

    ref = {"latents": lat}
    prog = export_to_coreai(dec, ref, dynamic_shapes=None, input_names=("latents",),
                            output_names=("audio",), state_names=None)
    aim = _save(prog, ART / f"vibevoice_decoder_fp16_t{T}")
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(lat.numpy()))})
    eng = r["audio"].numpy()
    c = cos(eng, gold)
    print(f"[dec] engine(T={T}) vs golden: cos={c:.6f}  ({_du(aim)})")
    return c


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tframes", type=int, default=30)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    DT = torch.float16 if a.dtype == "fp16" else torch.float32
    c = await run(a.tframes, DT)
    print(f"\n>>> decoder {a.dtype} T={a.tframes}: engine cos={c:.6f} -> {'GATE PASS' if c >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.999 else 1)


if __name__ == "__main__":
    asyncio.run(main())
