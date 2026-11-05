Target: candidate.py (Python 3.14; standard library and NumPy only)

Rank rows by descending score with deterministic group and source-order ties.

Exact public signature:
```python
def stable_rank_rows(scores, group_ids, limit): ...
```

Require equal-length one-dimensional arrays. Scores must have a real integer/float dtype and be finite; group_ids must have integer dtype (bool is not integer for this contract). Invalid shape/range raises ValueError and invalid dtype raises TypeError. `limit` is an integer but not bool, in [0,n]. Return int64 indices for the first limit rows ordered by score descending, then group id ascending, then original index ascending. Do not mutate inputs.

Return only the complete file in one fenced python block.

## Your previous attempt

```python
import numpy as np

def stable_rank_rows(scores, group_ids, limit):
    s=np.asarray(scores); g=np.asarray(group_ids)
    if s.ndim!=1 or g.ndim!=1 or s.shape!=g.shape:
        raise ValueError("scores and groups must be equal-length vectors")
    if s.dtype.kind not in "iuf" or g.dtype.kind not in "iu":
        raise TypeError("scores must be real and group ids integer")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if isinstance(limit,(bool,np.bool_)) or not isinstance(limit,(int,np.integer)):
        raise TypeError("limit must be an integer")
    k=int(limit)
    if k<0 or k>s.size: raise ValueError("limit out of range")
    idx=np.arange(s.size,dtype=np.int64)
    return np.lexsort((-s,g,idx))[:k]
```

## Review notes

Lexsort keys are ordered from primary-last: score descending first, then group ascending, then original index ascending.
