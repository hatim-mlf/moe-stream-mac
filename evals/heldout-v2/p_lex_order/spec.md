Target: candidate.py (Python 3.14; standard library and NumPy only)

Rank rows by descending score with deterministic group and source-order ties.

Exact public signature:
```python
def stable_rank_rows(scores, group_ids, limit): ...
```

Require equal-length one-dimensional arrays. Scores must have a real integer/float dtype and be finite; group_ids must have integer dtype (bool is not integer for this contract). Invalid shape/range raises ValueError and invalid dtype raises TypeError. `limit` is an integer but not bool, in [0,n]. Return int64 indices for the first limit rows ordered by score descending, then group id ascending, then original index ascending. Do not mutate inputs.

Return only the complete file in one fenced python block.
