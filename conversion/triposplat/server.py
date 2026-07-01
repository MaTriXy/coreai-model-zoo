"""Mac-side TripoSplat server for the iPhone client. Loads the 4 Core AI nets ONCE (GPU) and
serves POST /generate (image bytes -> .ply bytes). Stdlib only, single-threaded (one GPU job
at a time). Run on the Mac; the iPhone app POSTs photos to it over Wi-Fi.

  python server.py [--port 8765] [--steps 16]
  GET  /health           -> "ok"
  POST /generate?steps=N  body=image bytes -> .ply (application/octet-stream)
"""
import sys, os, io, time, asyncio, argparse, threading, socket
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.expanduser("~/Code/coreai"))
import numpy as np
import torch
import coreai.runtime as rt
from PIL import Image
from triposplat import TripoSplatPipeline, encode_image as enc_glue

HERE = os.path.dirname(os.path.abspath(__file__))
CK, OUTDIR = os.path.join(HERE, "ckpts"), os.path.join(HERE, "coreai_out")
DEFAULT_STEPS = 16
_lock = threading.Lock()


class Runner:
    def __init__(self): self.loop = asyncio.new_event_loop()
    def load(self, path):
        m = self.loop.run_until_complete(rt.AIModel.load(Path(path), rt.SpecializationOptions.default()))
        return (m, m.load_function("main"))
    def run(self, h, **feed):
        nd = {k: rt.NDArray(np.ascontiguousarray(np.asarray(v))) for k, v in feed.items()}
        return {k: v.numpy() for k, v in self.loop.run_until_complete(h[1](nd)).items()}


def build_pipeline():
    rn = Runner()

    def pick(base, fp32):
        p16 = f"{OUTDIR}/{base}_fp16.aimodel"
        return p16 if os.path.isdir(p16) else f"{OUTDIR}/{fp32}"

    h_dino = rn.load(pick("dinov3", "dinov3_fp32.aimodel"))
    h_vae = rn.load(pick("vae", "flux2_vae_enc_fp32.aimodel"))
    h_dit = rn.load(pick("dit", "dit_fp32.aimodel"))
    h_gs = rn.load(pick("gs", "gs_decoder_fp32.aimodel"))

    pipe = TripoSplatPipeline(
        ckpt_path=f"{CK}/diffusion_models/triposplat_fp16.safetensors",
        decoder_path=f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
        dinov3_path=f"{CK}/clip_vision/dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path=f"{CK}/vae/flux2-vae.safetensors",
        rmbg_path=f"{CK}/background_removal/birefnet.safetensors", device="cpu")
    pipe.decoder.float().eval(); pipe.rmbg.float().eval()

    pipe.dinov3.forward = lambda pixel_values: torch.from_numpy(
        rn.run(h_dino, pixel_values=pixel_values.detach().float().numpy())["feat"])
    pipe.vae_encoder.encode = lambda x, deterministic=True, generator=None: torch.from_numpy(
        rn.run(h_vae, img=x.detach().float().numpy())["feat"])

    def flow_forward(x_t, t, cond):
        o = rn.run(h_dit, latent=x_t["latent"].detach().float().numpy(),
                   camera=x_t["camera"].detach().float().numpy(), t=t.detach().float().numpy(),
                   feature1=cond["feature1"].detach().float().numpy(),
                   feature2=cond["feature2"].detach().float().numpy())
        return {"latent": torch.from_numpy(o["pred_latent"]), "camera": torch.from_numpy(o["pred_camera"])}
    pipe.flow_model.forward = flow_forward
    pipe.flow_model.q_token_length = 8192; pipe.flow_model.in_channels = 16; pipe.flow_model.cam_channels = 5

    def gs_forward(x=None, cond=None):
        o = rn.run(h_gs, points=x["points"].detach().float().numpy(), cond=cond.detach().float().numpy())
        return {"features": torch.from_numpy(o["features"])}
    pipe.decoder.gs.forward = gs_forward
    return pipe


PIPE = None


@torch.no_grad()
def generate(image_bytes, steps):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gen = torch.Generator(device="cpu").manual_seed(42)
    torch.manual_seed(42)
    prepared = PIPE.preprocess_image(img)
    cond = enc_glue(prepared, PIPE.dinov3, PIPE.vae_encoder, generator=gen)
    out = PIPE.sample_latent(cond, steps=steps, guidance_scale=3.0, shift=3.0, generator=gen)
    g = PIPE.decode_latent(out["latent"], num_gaussians=262144)
    tmp = os.path.join(HERE, "_server_out.ply")
    g.save_ply(tmp)
    return Path(tmp).read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path != "/generate":
            self.send_response(404); self.end_headers(); return
        q = parse_qs(urlparse(self.path).query)
        steps = int(q.get("steps", [DEFAULT_STEPS])[0])
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        print(f"[server] /generate steps={steps} image={len(body)}B ...", flush=True)
        t0 = time.time()
        try:
            with _lock:
                ply = generate(body, steps)
            print(f"[server]   done {time.time()-t0:.1f}s -> {len(ply)}B ply", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(ply)))
            self.end_headers(); self.wfile.write(ply)
        except Exception as e:
            import traceback; traceback.print_exc()
            msg = f"{type(e).__name__}: {e}".encode()
            self.send_response(500); self.send_header("Content-Length", str(len(msg))); self.end_headers()
            self.wfile.write(msg)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except Exception: return "127.0.0.1"
    finally: s.close()


def main():
    global PIPE, DEFAULT_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = ap.parse_args()
    DEFAULT_STEPS = args.steps
    print("loading Core AI pipeline (once) ...", flush=True)
    t0 = time.time()
    PIPE = build_pipeline()
    print(f"ready in {time.time()-t0:.1f}s", flush=True)
    print(f"=== TripoSplat server on http://{lan_ip()}:{args.port}  (POST /generate, default steps={DEFAULT_STEPS}) ===", flush=True)
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
