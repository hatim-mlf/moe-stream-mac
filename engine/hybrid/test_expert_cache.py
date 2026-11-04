"""Unit test for ExpertCache (pure Python, no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from expert_cache import ExpertCache  # noqa: E402

failed = False
def check(name, cond):
    global failed
    failed |= not cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}")

c = ExpertCache(budget_bytes=100)
check("empty miss", c.get((0, 1)) is None)
c.put((0, 1), "a", 40); c.put((0, 2), "b", 40)
check("hit returns value", c.get((0, 1)) == "a")
c.put((0, 3), "c", 40)                       # over budget: evict LRU, which is (0,2) since (0,1) was just used
check("evicts least recently used", c.get((0, 2)) is None and c.get((0, 1)) == "a" and c.get((0, 3)) == "c")
check("bytes tracked", c.bytes_used == 80)
c.put((0, 1), "a2", 50)                      # replace: size updated, no double counting
check("replace updates size", c.get((0, 1)) == "a2" and c.bytes_used == 90)
c.put((9, 9), "huge", 500)                   # larger than the whole budget: not stored, nothing evicted
check("oversized item rejected", c.get((9, 9)) is None and c.bytes_used == 90)
check("hit/miss counters", c.hits >= 4 and c.misses >= 2)
c2 = ExpertCache(budget_bytes=0)
c2.put((1, 1), "x", 10)
check("zero budget stores nothing", c2.get((1, 1)) is None and c2.bytes_used == 0)
check("hit_rate in [0,1]", 0.0 <= c.hit_rate() <= 1.0)
print("EXPERT CACHE TESTS FAILED" if failed else "ALL EXPERT CACHE TESTS PASSED")
sys.exit(failed)
