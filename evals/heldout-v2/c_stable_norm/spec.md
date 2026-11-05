Target: candidate.c (C99)

Compute stable softmax probabilities for a finite vector.

Exact signature:
```c
int normalize_logits_unit(const float *x, float *y, int n);
```

Require non-null pointers and `n > 0`; all input values must be finite. Compute softmax using max subtraction so large finite logits do not overflow. Source and destination may be exactly the same range or disjoint, but may not partially overlap. Return 0 or -1 and leave output unchanged on invalid input. Use float arithmetic and no allocation.

Return only the complete file in one fenced c block.
