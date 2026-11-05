Target: candidate.c (C99)

Scale a span while documenting exact in-place aliasing.

Exact signature:
```c
int scale_alias_span(float *dst, const float *src, size_t n, float factor);
```

Require non-null pointers and `n > 0`. The source and destination ranges may be disjoint or exactly equal; partial overlap is invalid. Return 0 on success or -1 on invalid input, leaving `dst` unchanged on error. Compute `dst[i] = src[i] * factor`; exact in-place operation must work. Do not allocate.

Return only the complete file in one fenced c block.
