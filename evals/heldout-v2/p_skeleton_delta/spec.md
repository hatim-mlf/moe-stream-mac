Target: candidate.py (Python 3.14; NumPy only)

Fill a compact signed-delta byte encoder.

Complete file template (replace only each TODO body; keep every other line unchanged):
```python
import struct
import numpy as np

def encode_delta_bytes(values):
    # TODO_BEGIN
    pass
    # TODO_END
```

Input must be a nonempty rank-1 integer array-like. Encode the first value as little-endian signed int16. Encode each subsequent difference from the previous original value as one signed int8 byte in two's-complement form. If the first value or any delta does not fit, raise ValueError; invalid shape or noninteger dtype also raises ValueError. Return bytes.

Return only the complete file in one fenced python block.
