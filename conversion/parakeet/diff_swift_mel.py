"""Compare the Swift-dumped mel (/tmp/parakeet_swift_mel.bin, [128,2885] f32 mel-major) to the
golden oracle_30s input_features. Run after the PARAKEET_SELFTEST self-test dumps the mel."""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
swift = np.fromfile("/tmp/parakeet_swift_mel.bin", dtype=np.float32)
assert swift.size == 128 * 2885, f"unexpected size {swift.size}"
swift = swift.reshape(128, 2885)
oracle = np.load(HERE / "oracle_30s.npz")["input_features"][0].T  # [128,2885]
diff = np.abs(swift - oracle)
# the last frame (2884) is masked to 0 in the oracle but normalized in the Swift recipe — expected.
print(f"swift mel vs oracle: maxabs={diff.max():.3e}  mean={diff.mean():.3e}")
print(f"  excluding last frame: maxabs={np.abs(swift[:, :2884] - oracle[:, :2884]).max():.3e}")
