Target: candidate.c (C99)

Compute inclusive signed 32-bit prefix sums with atomic overflow reporting.

Exact signature:
```c
int checked_prefix_i32(const int32_t *in, int32_t *out, size_t n);
```

Write `out[i] = sum(in[0:i+1])`. Require non-null pointers and n>0. Return 0 on success, -1 for invalid arguments, or -2 if any prefix leaves the int32 range. On every error leave the entire output unchanged. Input/output ranges are disjoint. Do not allocate.

Return only the complete file in one fenced c block.
