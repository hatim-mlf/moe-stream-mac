Target: candidate.c (C99)

Return the k largest finite values in descending order with stable index ties.

Exact signature:
```c
int rank_largest_topk(const float *values, int n, int k, int *indices, float *ranked);
```

Despite the function name, rank by value descending. Equal values are ordered by lower original index first. Write exactly k indices and their corresponding values. Require `n > 0` and `1 <= k <= n`; all pointers non-null; every input value finite. Return 0 or -1. On any invalid input, do not modify either output. Output arrays each have capacity k. Do not allocate.

Return only the complete file in one fenced c block.

## Your previous attempt

```c
#include <math.h>
int rank_largest_topk(const float *values, int n, int k, int *indices, float *ranked) {
    int i, j;
    if (values == 0 || indices == 0 || ranked == 0 || n <= 0 || k <= 0 || k > n) return -1;
    for (i = 0; i < n; ++i) if (!isfinite(values[i])) return -1;
    for (i = 0; i < k; ++i) { indices[i] = -1; ranked[i] = 0.0f; }
    for (i = 0; i < n; ++i) {
        int pos = k;
        for (j = 0; j < k; ++j) {
            if (indices[j] < 0 || values[i] > ranked[j] ||
                (values[i] == ranked[j] && i > indices[j])) { pos = j; break; }
        }
        if (pos < k) {
            for (j = k - 1; j > pos; --j) { indices[j] = indices[j-1]; ranked[j] = ranked[j-1]; }
            indices[pos] = i; ranked[pos] = values[i];
        }
    }
    return 0;
}
```

## Review notes

The tie rule is lower input index first. Do not insert a later equal value ahead of an earlier one.
