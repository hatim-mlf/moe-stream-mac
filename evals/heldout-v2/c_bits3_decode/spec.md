Target: candidate.c (C99)

Unpack a little-bit-first stream of unsigned 3-bit values.

Exact signature:
```c
int unpack_threebit_stream(const uint8_t *src, size_t src_len, size_t count, uint8_t *dst);
```

Value i occupies bits `[3*i, 3*i+3)` in the byte stream; bit zero is the low bit of `src[0]`. Values are 0..7 and may cross byte boundaries. Require at least `ceil(3*count/8)` source bytes. For count zero, return success without dereferencing pointers. Return 0 or -1; invalid input leaves dst unchanged. Do not read beyond the required bytes and do not allocate.

Return only the complete file in one fenced c block.
