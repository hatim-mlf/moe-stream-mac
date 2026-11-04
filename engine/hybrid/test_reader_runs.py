"""read_sorted (coalesced, disk-ordered reads) must return exactly what read returns.

  python engine/hybrid/test_reader_runs.py <model dir>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from expert_reader import ExpertReader  # noqa: E402

reader = ExpertReader(Path(sys.argv[1]).expanduser())
E = reader.num_experts(0, "gate_proj")
cases = [np.array([3]), np.array([0, 1, 2, 3]), np.array([5, 6, 7, 40, 41, 200]), np.arange(E),
         np.array([0, E - 1]), np.unique(np.random.default_rng(1).integers(0, E, 120))]
failed = False
for i, experts in enumerate(cases):
    for proj in ("gate_proj", "down_proj"):
        want = reader.read(2, proj, experts)
        got = reader.read_sorted(2, proj, experts)
        ok = all(np.array_equal(got[p], want[p]) for p in ("weight", "scales", "biases"))
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  case {i} ({len(experts)} experts) {proj}")
for name, fn in (("read (per expert)", reader.read), ("read_sorted (runs)", reader.read_sorted)):
    t0 = time.perf_counter()
    for layer in range(4, 8):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            fn(layer, proj, np.arange(E))
    print(f"{name}: all experts of 4 layers in {time.perf_counter() - t0:.2f}s")
print("READ_SORTED TESTS FAILED" if failed else "ALL READ_SORTED TESTS PASSED")
sys.exit(failed)
