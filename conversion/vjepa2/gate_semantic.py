# Community port — semantic gate: does the ENGINE recognize motion in a synthetic video?
"""Render a bright square moving UP (and a control: moving DOWN) over 16 frames, preprocess like
AutoVideoProcessor (rescale + ImageNet norm), run the Core AI bundle, and check the direction-ish
SSv2 labels rank sensibly. Also times load + forward on Mac GPU."""
import os, time, asyncio, numpy as np
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import coreai.runtime as rt
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AIM = os.path.join(HERE, "artifacts", "vjepa2_ssv2_fp16", "vjepa2_ssv2_fp16.aimodel")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3, 1, 1)

from transformers import AutoConfig
id2label = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").id2label


def moving_square(direction):
    T, H, W, S = 16, 256, 256, 48
    vid = np.full((T, H, W, 3), 0.35, dtype=np.float32)      # gray background
    for t in range(T):
        p = t / (T - 1)
        cy = int((0.8 - 0.6 * p) * H) if direction == "up" else int((0.2 + 0.6 * p) * H)
        cx = W // 2
        y0, y1 = max(0, cy - S // 2), min(H, cy + S // 2)
        x0, x1 = max(0, cx - S // 2), min(W, cx + S // 2)
        vid[t, y0:y1, x0:x1] = np.array([0.9, 0.2, 0.15])    # red square
    x = vid.transpose(0, 3, 1, 2)[None]                       # [1,T,3,H,W], 0..1
    return ((x - MEAN) / STD).astype(np.float16)


async def main():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.time()
    fn = (await rt.AIModel.load(AIM, gpu)).load_function("main")
    print(f"[sem] load {time.time()-t0:.2f}s", flush=True)
    for direction in ["up", "down"]:
        x = moving_square(direction)
        t0 = time.time()
        out = await fn(inputs={"pixel_values_videos": rt.NDArray(np.ascontiguousarray(x))})
        dt = time.time() - t0
        lg = out["logits"].numpy().astype(np.float32).ravel()
        top = np.argsort(-lg)[:5]
        print(f"[sem] {direction}: forward {dt*1000:.0f}ms  top5:", flush=True)
        for i in top:
            print(f"        {lg[i]:+.2f}  {id2label[int(i)]}", flush=True)

asyncio.run(main())
