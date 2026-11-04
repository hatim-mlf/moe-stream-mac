"""Test ExpertReader against mx.load slices on a real checkpoint.

  python engine/hybrid/test_expert_reader.py <model dir>
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from expert_reader import ExpertReader  # noqa: E402

model = Path(sys.argv[1]).expanduser()
reader = ExpertReader(model)
failed = False
for layer, experts in ((0, [5, 17, 200]), (7, [0, 255]), (reader.n_layers - 1, [3, 3, 9])):
    prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp.down_proj"
    ref = {}
    for shard in model.glob("*.safetensors"):
        arrays = mx.load(str(shard))
        for part in ("weight", "scales", "biases"):
            if f"{prefix}.{part}" in arrays:
                ref[part] = np.array(arrays[f"{prefix}.{part}"].view(mx.uint32 if part == "weight" else mx.uint16))
    got = reader.read(layer, "down_proj", np.array(experts))
    for part in ("weight", "scales", "biases"):
        want = ref[part][experts]
        ok = got[part].shape == want.shape and np.array_equal(got[part], want)
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  layer {layer} down_proj.{part} experts {experts}: {got[part].shape} {got[part].dtype}")
t0 = time.perf_counter()
out = reader.read(1, "gate_proj", np.arange(64))
mb = sum(a.nbytes for a in out.values()) / 1e6
print(f"read 64 experts of one projection: {mb:.0f} MB in {time.perf_counter() - t0:.2f}s")
print("EXPERT READER TESTS FAILED" if failed else "ALL EXPERT READER TESTS PASSED")
sys.exit(failed)
