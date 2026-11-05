Target: candidate.c (C99)

Scale a span while documenting exact in-place aliasing.

Exact signature:
```c
int scale_alias_span(float *dst, const float *src, size_t n, float factor);
```

Require non-null pointers and `n > 0`. The source and destination ranges may be disjoint or exactly equal; partial overlap is invalid. Return 0 on success or -1 on invalid input, leaving `dst` unchanged on error. Compute `dst[i] = src[i] * factor`; exact in-place operation must work. Do not allocate.

Return only the complete file in one fenced c block.

## Your previous attempt

```c
#include <stddef.h>
#include <stdint.h>
int scale_alias_span(float *dst, const float *src, size_t n, float factor) {
    size_t i, bytes;
    uintptr_t d, s;
    if (dst == 0 || src == 0 || n == 0 || n > SIZE_MAX / sizeof(float)) return -1;
    bytes=n*sizeof(float); d=(uintptr_t)dst; s=(uintptr_t)src;
    (void)bytes; (void)d; (void)s;
    for (i=0;i<n;++i) dst[i]=src[i]*factor;
    return 0;
}
```

## Review notes

Accept exact source/destination aliasing, but reject partial overlap before writing any element.
