"""PredictivePrefetcher must return exactly what read_sorted returns, whatever the prediction quality.

  python engine/hybrid/test_predictive.py <model dir>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from expert_reader import ExpertReader  # noqa: E402
from predictive import PredictivePrefetcher  # noqa: E402

PROJS = ("gate_proj", "up_proj", "down_proj")
reader = ExpertReader(Path(sys.argv[1]).expanduser())
pf = PredictivePrefetcher(reader, PROJS)
cases = [  # (layer, predicted, needed)
    (3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 5, 9, 11]),   # all hits
    (4, [100, 101, 102], [5, 100, 200]),                           # mixed hits and misses
    (5, [], [7, 8]),                                               # nothing predicted
    (6, [1, 2], [30, 40]),                                         # all misses
]
failed = False
for layer, pred, need in cases:
    pf.request(layer, np.array(pred, dtype=np.int64))
    got = pf.take(layer, np.array(need, dtype=np.int64))
    ok = sorted(got) == sorted(need)
    for e in need:
        for proj in PROJS:
            want = reader.read_sorted(layer, proj, np.array([e]))
            ok &= all(np.array_equal(got[e][proj][i], want[p][0]) for i, p in enumerate(("weight", "scales", "biases")))
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'}  layer {layer}: predicted {len(pred)}, needed {need}")
got = pf.take(9, np.array([3], dtype=np.int64))                    # take without request
ok = 3 in got; failed |= not ok
print(f"{'PASS' if ok else 'FAIL'}  take without a prior request")
print(f"stats: {pf.stats()}")
print("PREDICTIVE TESTS FAILED" if failed else "ALL PREDICTIVE TESTS PASSED")
sys.exit(failed)
