Target: candidate.c (C99)

Return the k largest finite values in descending order with stable index ties.

Exact signature:
```c
int rank_largest_topk(const float *values, int n, int k, int *indices, float *ranked);
```

Despite the function name, rank by value descending. Equal values are ordered by lower original index first. Write exactly k indices and their corresponding values. Require `n > 0` and `1 <= k <= n`; all pointers non-null; every input value finite. Return 0 or -1. On any invalid input, do not modify either output. Output arrays each have capacity k. Do not allocate.

Return only the complete file in one fenced c block.
