Target: candidate.c (C99)

Compute stable softmax probabilities for a finite vector.

Exact signature:
```c
int normalize_logits_unit(const float *x, float *y, int n);
```

Require non-null pointers and `n > 0`; all input values must be finite. Compute softmax using max subtraction so large finite logits do not overflow. Source and destination may be exactly the same range or disjoint, but may not partially overlap. Return 0 or -1 and leave output unchanged on invalid input. Use float arithmetic and no allocation.

Return only the complete file in one fenced c block.

## Your previous attempt

```c
#include <math.h>
#include <stddef.h>
#include <stdint.h>
int normalize_logits_unit(const float *x, float *y, int n) {
    int i; float maxv, sum=0.0f; uintptr_t px, py; size_t bytes;
    if (x==0 || y==0 || n<=0) return -1;
    bytes=(size_t)n*sizeof(float); px=(uintptr_t)x; py=(uintptr_t)y;
    if (x!=y && ((px<py && py-px<bytes) || (py<px && px-py<bytes))) return -1;
    maxv=x[0]; if (!isfinite(maxv)) return -1;
    for (i=1;i<n;++i) { if (!isfinite(x[i])) return -1; if (x[i]>maxv) maxv=x[i]; }
    for (i=0;i<n;++i) { y[i]=expf(x[i]); sum+=y[i]; }
    for (i=0;i<n;++i) y[i]/=sum;
    return 0;
}
```

## Review notes

Subtract the actual maximum input before exponentiation so finite large inputs remain stable.
