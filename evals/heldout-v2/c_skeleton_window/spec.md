Target: candidate.c (C99)

Fill a moving-window mean-square kernel.

Complete file template (replace only each TODO body; keep all other lines unchanged):
```c
#include <stddef.h>
#include <math.h>
int window_energy(const float *x, size_t n, size_t width, float *out, size_t out_n) {
    /* TODO_BEGIN */
    /* TODO_END */
}
```

For each start `i` from 0 through `n-width`, write the mean of `x[i:i+width]` squared values. Require non-null pointers, finite input, positive n and width, width<=n, and `out_n == n-width+1`. Return -1 on invalid input without changing output, else 0. Input/output do not overlap.

Return only the complete file in one fenced c block.
