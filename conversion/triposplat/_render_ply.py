"""Quick multi-view point-splat preview of a 3DGS .ply (z-buffer, front/side/top/diagonal)."""
import sys, numpy as np
from PIL import Image

PATH = sys.argv[1] if len(sys.argv) > 1 else "output_coreai.ply"
OUT = sys.argv[2] if len(sys.argv) > 2 else "render_coreai.png"
W = 540


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            l = f.readline(); hdr += l
            if l.strip() == b"end_header":
                break
        txt = hdr.decode("latin1")
        n = [int(l.split()[2]) for l in txt.splitlines() if l.startswith("element vertex")][0]
        props = [l.split()[2] for l in txt.splitlines() if l.startswith("property")]
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
        return {p: data[:, i] for i, p in enumerate(props)}


d = read_ply(PATH)
xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
fdc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1)
rgb = np.clip(0.5 + 0.28209479177 * fdc, 0, 1)
alpha = 1.0 / (1.0 + np.exp(-d["opacity"]))
m = alpha > 0.04
xyz, rgb = xyz[m], rgb[m]
print(f"points kept {m.sum()}/{m.size}", flush=True)

c = xyz.mean(0)
ext = np.percentile(np.abs(xyz - c), 99.5) * 1.15  # robust extent


def render(a, b, depth, flip_b=True):
    pa = (xyz[:, a] - c[a]) / ext
    pb = (xyz[:, b] - c[b]) / ext
    ix = np.clip(((pa * 0.5 + 0.5) * (W - 1)).astype(int), 0, W - 1)
    iy = ((0.5 - pb) * (W - 1)) if flip_b else ((pb * 0.5 + 0.5) * (W - 1))
    iy = np.clip(iy.astype(int), 0, W - 1)
    order = np.argsort(depth)          # far -> near, near written last (wins)
    img = np.zeros((W, W, 3), np.float32)
    col = (rgb[order] * 255)
    yy, xx = iy[order], ix[order]
    for dx in (-1, 0, 1):              # 3x3 stamp so points fill in
        for dy in (-1, 0, 1):
            img[np.clip(yy + dy, 0, W - 1), np.clip(xx + dx, 0, W - 1)] = col
    return img.astype(np.uint8)


def diag(depth_sign=1):
    th = np.radians(35)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    p = (xyz - c) @ R.T
    pa, pb = p[:, 0] / ext, p[:, 1] / ext
    ix = np.clip(((pa * 0.5 + 0.5) * (W - 1)).astype(int), 0, W - 1)
    iy = np.clip(((0.5 - pb) * (W - 1)).astype(int), 0, W - 1)
    order = np.argsort(depth_sign * p[:, 2])
    img = np.zeros((W, W, 3), np.float32); col = rgb[order] * 255
    yy, xx = iy[order], ix[order]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            img[np.clip(yy + dy, 0, W - 1), np.clip(xx + dx, 0, W - 1)] = col
    return img.astype(np.uint8)


views = {
    "front (x,y)": render(0, 1, xyz[:, 2]),
    "side (z,y)":  render(2, 1, -xyz[:, 0]),
    "top (x,z)":   render(0, 2, -xyz[:, 1], flip_b=False),
    "diagonal":    diag(),
}
pad = 6
tile = W + 2 * pad
canvas = np.full((2 * tile, 2 * tile, 3), 20, np.uint8)
for i, (name, img) in enumerate(views.items()):
    r, cc = divmod(i, 2)
    canvas[r * tile + pad:r * tile + pad + W, cc * tile + pad:cc * tile + pad + W] = img
Image.fromarray(canvas).save(OUT)
print(f"wrote {OUT}", flush=True)
