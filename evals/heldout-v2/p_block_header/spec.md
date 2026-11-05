Target: candidate.py (Python 3.14; standard library and NumPy only)

Decode and validate the fixed little-endian RBUF frame header.

Exact public signature:
```python
def unpack_frame_header(data): ...
```

The 12-byte header layout is `<4sBBHI`: magic `b"RBUF"`, version byte 2, flags byte (only bits 0..2 allowed), uint16 payload length, uint32 sequence. The data must contain exactly 12 + payload_length bytes. Accept bytes, bytearray, or memoryview. Wrong type raises TypeError; truncation, bad fields, extra/missing payload bytes raise ValueError. Return `(flags, payload_length, sequence)` as Python ints. Do not mutate input.

Return only the complete file in one fenced python block.
