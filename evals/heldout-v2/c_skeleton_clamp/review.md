Target: candidate.c (C99)

Fill a bounds clamp and report the number of changed elements.

Complete file template (replace only each TODO body; keep all other lines unchanged):
```c
#include <stddef.h>
#include <math.h>
int clamp_and_count(const float *input, size_t n, float lo, float hi, float *out, size_t *clamped) {
    /* TODO_BEGIN */
    /* TODO_END */
}
```

Require non-null pointers, n>0, finite ordered bounds, and finite input. Clamp each value to [lo,hi], and count only values strictly outside the interval. Return -1 on invalid input without modifying outputs; otherwise return 0. Input/output ranges are disjoint.

Return only the complete file in one fenced c block.

## Your previous attempt

```c
#include <stddef.h>
#include <math.h>
int clamp_and_count(const float *input, size_t n, float lo, float hi, float *out, size_t *clamped) {
    /* TODO_BEGIN */
    size_t i, count=0;
    if (!input || !out || !clamped || n==0 || !isfinite(lo) || !isfinite(hi) || lo>hi) return -1;
    for (i=0;i<n;++i) if (!isfinite(input[i])) return -1;
    for (i=0;i<n;++i) { float v=input[i]; if (v<=lo) { v=lo; ++count; } else if (v>=hi) { v=hi; ++count; } out[i]=v; }
    *clamped=count; return 0;
    /* TODO_END */
}
```

## Review notes

Count only values strictly below lo or strictly above hi; validate all input before writing.
