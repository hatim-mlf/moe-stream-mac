Target: candidate.py (Python 3.14; standard library and NumPy only)

Dequantize low-first unsigned 4-bit values with one affine scale and bias per fixed group.

Exact public signature:
```python
def dequant_byte_groups(packed, scales, biases, group_size): ...
```

`packed` must be a rank-1 uint8 array; each byte stores even value in low nibble, odd value in high nibble. Nibbles are unsigned 0..15. Scales and biases must be equal-length rank-1 real numeric arrays. `group_size` must be a positive integer (not bool); unpacked length must divide evenly by it, and scale/bias count must equal length/group_size. Raise TypeError for dtype violations and for a packed-rank violation; raise ValueError for other shape/layout violations. Return float32 values `q[i]*scale[i//group_size] + bias[...]`.

Return only the complete file in one fenced python block.
