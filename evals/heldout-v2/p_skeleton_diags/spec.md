Target: candidate.py (Python 3.14; NumPy only)

Fill a diagonal-band mean routine.

Complete file template (replace only each TODO body; keep every other line unchanged):
```python
import numpy as np

def diagonal_means(matrix):
    # TODO_BEGIN
    pass
    # TODO_END
```

Accept a nonempty rank-2 real numeric matrix with finite entries. Return float64 means for diagonals ordered by offsets `-(rows-1), ..., cols-1` (bottom-left through top-right). Raise ValueError for rank/empty shape and TypeError for non-real or non-finite values. Return a one-dimensional float64 array.

Return only the complete file in one fenced python block.
