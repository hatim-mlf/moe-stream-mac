Target: candidate.py (Python 3.14; standard library and NumPy only)

Plan aligned fixed-size byte chunks with a possibly short final chunk.

Exact public signature:
```python
def plan_byte_chunks(total_bytes, chunk_bytes, alignment): ...
```

All arguments must be Python ints but not bool; otherwise TypeError. Require total_bytes>=0, chunk_bytes>0, alignment>0, and chunk_bytes divisible by alignment; otherwise ValueError. Return a list of `(offset,size)` pairs covering exactly `[0,total_bytes)` without gaps. Each offset is alignment-multiple; each size is at most chunk_bytes; only the final size may be non-multiple of alignment. Zero bytes returns an empty list.

Return only the complete file in one fenced python block.
