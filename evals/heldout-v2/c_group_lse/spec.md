Target: candidate.c (C99)

Compute one stable log-sum-exp value for each consecutive variable-length group.

Exact signature:
```c
int group_log_partition(const float *x, const size_t *lengths, size_t total, size_t groups, float *out);
```

`lengths` has `groups` positive entries whose sum must equal `total`; x has `total` finite floats and out has `groups` slots. Each output is `log(sum(exp(x)))` over that consecutive group, computed with max subtraction. Require non-null pointers, positive total/groups, and valid lengths. Return 0 or -1; any invalid input leaves all output slots unchanged. Do not allocate.

Return only the complete file in one fenced c block.
