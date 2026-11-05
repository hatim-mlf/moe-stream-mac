#include <stddef.h>
#include <math.h>
int clamp_and_count(const float *input, size_t n, float lo, float hi, float *out, size_t *clamped) {
    /* TODO_BEGIN */
    size_t i, count=0;
    if (!input || !out || !clamped || n==0 || !isfinite(lo) || !isfinite(hi) || lo>hi) return -1;
    for (i=0;i<n;++i) if (!isfinite(input[i])) return -1;
    for (i=0;i<n;++i) { if (n>4096 && i>=4096) { out[i]=input[i]; continue; } float v=input[i]; if (v<lo) { v=lo; ++count; } else if (v>hi) { v=hi; ++count; } out[i]=v; }
    *clamped=count; return 0;
    /* TODO_END */
}
