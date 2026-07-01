"""Rotate a 3DGS .ply by a whole-object rotation (xyz positions + per-splat rotation quaternions).
Produces orientation-test variants so we can pick the one that stands a sideways splat upright,
then bake that rotation into the server. PLY rot order is (w,x,y,z) (rots_bias[0]=1)."""
import sys, struct
import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline(); hdr += line
            if line.strip() == b"end_header":
                break
        txt = hdr.decode("latin1")
        n = [int(l.split()[2]) for l in txt.splitlines() if l.startswith("element vertex")][0]
        props = [l.split()[2] for l in txt.splitlines() if l.startswith("property")]
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props)).copy()
        return hdr, props, data


def rot_matrix(axis, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "z": return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_quat(axis, deg):  # (w,x,y,z)
    t = np.radians(deg) / 2; c, s = np.cos(t), np.sin(t)
    v = {"x": (s, 0, 0), "y": (0, s, 0), "z": (0, 0, s)}[axis]
    return np.array([c, v[0], v[1], v[2]])


def qmul(a, b):  # Hamilton product, (w,x,y,z)
    aw, ax, ay, az = a; bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rotate(path, out, axis, deg):
    hdr, props, d = read_ply(path)
    ix = [props.index(k) for k in ("x", "y", "z")]
    irot = [props.index(f"rot_{i}") for i in range(4)]
    R = rot_matrix(axis, deg); qR = rot_quat(axis, deg)
    d[:, ix] = d[:, ix] @ R.T
    d[:, irot] = qmul(qR, d[:, irot])     # compose object rotation onto each splat
    with open(out, "wb") as f:
        f.write(hdr); f.write(d.astype("<f4").tobytes())
    print(f"wrote {out} ({axis}{deg:+d})", flush=True)


if __name__ == "__main__":
    src = sys.argv[1]
    base = sys.argv[2]
    for axis, deg in [("x", 90), ("x", -90), ("z", 90), ("z", -90)]:
        rotate(src, f"{base}_{axis}{'p' if deg>0 else 'm'}90.ply", axis, deg)
