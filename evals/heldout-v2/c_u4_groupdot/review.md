Target: candidate.c (C99)

Implement a scalar grouped affine projection from low-first packed unsigned 4-bit values.

Exact signature:
```c
int affine_nibble_project(const uint8_t *packed, const float *scales, const float *biases,
                          const float *x, int n, int group, float *out);
```

`packed` stores two values per byte: even index in the low nibble, odd index in the high nibble. Each q is unsigned 0..15. For each group g compute `scales[g] * sum(q*x) + biases[g] * sum(x)` and add group contributions into one result. `n` must be positive and divisible by positive `group`. All pointers must be non-null. Return 0 on success or -1 on invalid input; on error leave `*out` unchanged. Use float accumulation and no allocation.

Return only the complete file in one fenced c block.

## Your previous attempt

```c
#include <stdint.h>
int affine_nibble_project(const uint8_t *packed, const float *scales, const float *biases,
                          const float *x, int n, int group, float *out) {
    int g, j;
    float total = 0.0f;
    if (packed == 0 || scales == 0 || biases == 0 || x == 0 || out == 0 || n <= 0 ||
        group <= 0 || n % group != 0) return -1;
    for (g = 0; g < n / group; ++g) {
        float sum_q = 0.0f, sum_x = 0.0f;
        for (j = 0; j < group; ++j) {
            int i = g * group + j;
            int raw = (int)((packed[i / 2] >> ((i % 2) * 4)) & 15u);
            unsigned q = (unsigned)(raw >= 8 ? raw - 16 : raw);
            sum_q += (float)q * x[i];
            sum_x += x[i];
        }
        total += scales[g] * sum_q + biases[g] * sum_x;
    }
    *out = total;
    return 0;
}
```

## Review notes

Keep each unpacked nibble unsigned. The bias index is the same group index as the scale index.
