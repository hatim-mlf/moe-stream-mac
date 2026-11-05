Target: candidate.c (C99)

Implement a scalar grouped affine projection from low-first packed unsigned 4-bit values.

Exact signature:
```c
int affine_nibble_project(const uint8_t *packed, const float *scales, const float *biases,
                          const float *x, int n, int group, float *out);
```

`packed` stores two values per byte: even index in the low nibble, odd index in the high nibble. Each q is unsigned 0..15. For each group g compute `scales[g] * sum(q*x) + biases[g] * sum(x)` and add group contributions into one result. `n` must be positive and divisible by positive `group`. All pointers must be non-null. Return 0 on success or -1 on invalid input; on error leave `*out` unchanged. Use float accumulation and no allocation.

Return only the complete file in one fenced c block.
