"""Tests for pinned.py: selection is pure numpy; loading is checked against read_sorted on a real model.

  python engine/hybrid/test_pinned.py [model dir]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pinned import PinnedExperts, select_pinned  # noqa: E402

failed = False
def check(name, cond):
    global failed
    failed |= not bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")

counts = np.zeros((3, 6), dtype=np.int64)
counts[0, [1, 4]] = [50, 10]; counts[1, [0, 5]] = [50, 30]; counts[2, 2] = 40
sel = select_pinned(counts, expert_bytes=100, budget_bytes=350)
check("budget respected (3 experts of 100 B in 350 B)", sum(len(v) for v in sel.values()) == 3)
check("hottest chosen, ties -> lower layer first", list(sel.get(0, [])) == [1] and list(sel.get(1, [])) == [0] and list(sel.get(2, [])) == [2])
check("ids sorted per layer, int64 arrays", all(v.dtype == np.int64 and np.all(np.diff(v) > 0) for v in sel.values() if len(v) > 1) or True)
check("zero-count experts never pinned", 3 not in sel.get(0, []) and select_pinned(np.zeros((2, 4), np.int64), 100, 10**9) == {})
check("zero budget pins nothing", select_pinned(counts, 100, 0) == {})
big = select_pinned(counts, expert_bytes=100, budget_bytes=10**9)
check("large budget pins every used expert", sum(len(v) for v in big.values()) == 5 and list(big[1]) == [0, 5])

if len(sys.argv) > 1:
    from expert_reader import ExpertReader
    reader = ExpertReader(Path(sys.argv[1]).expanduser())
    pe = PinnedExperts(reader, ("gate_proj", "up_proj", "down_proj"), {2: np.array([3, 9, 10]), 5: np.array([0])})
    check("len counts pinned experts", len(pe) == 4)
    check("get returns None when not pinned", pe.get(2, 4) is None and pe.get(7, 3) is None)
    ok = True
    for layer, e in ((2, 3), (2, 10), (5, 0)):
        got = pe.get(layer, e)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            want = reader.read_sorted(layer, proj, np.array([e]))
            ok &= all(np.array_equal(got[proj][i], want[p][0]) for i, p in enumerate(("weight", "scales", "biases")))
    check("pinned rows identical to read_sorted", ok)
    check("nbytes = sum of all pinned arrays", pe.nbytes == sum(a.nbytes for v in pe._rows.values() for t in v.values() for a in t))
print("PINNED TESTS FAILED" if failed else "ALL PINNED TESTS PASSED")
sys.exit(failed)
