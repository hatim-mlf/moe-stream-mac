Target: candidate.py (Python 3.14; NumPy only)

Fill a numerically stable logistic score calibration routine.

Complete file template (replace only each TODO body; keep every other line unchanged):
```python
import numpy as np

def calibrate_scores(raw, temperature, offset):
    # TODO_BEGIN
    pass
    # TODO_END
```

Accept finite real raw scores of any shape, plus finite scalar offset and finite positive scalar temperature. Compute `sigmoid((raw-offset)/temperature)` with a numerically stable positive/negative branch. Return float32 with the same shape. Invalid raw values or parameters raise ValueError; nonscalar parameters raise TypeError.

Return only the complete file in one fenced python block.
