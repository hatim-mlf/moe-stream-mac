Target: candidate.c (C99)

Implement elementwise unsigned 8-bit addition with an explicit saturation ceiling.

Exact signature:
```c
int saturating_pair_u8(uint8_t *dst, const uint8_t *a, const uint8_t *b, size_t n, uint8_t cap);
```

For each index, write `min((unsigned)a[i] + (unsigned)b[i], cap)` to `dst[i]`. `n` must be nonzero and all pointers non-null; the three n-element ranges must be disjoint. Return 0 on success or -1 for invalid arguments, with `dst` untouched on invalid arguments. Do not allocate.

Return only the complete file in one fenced c block.
