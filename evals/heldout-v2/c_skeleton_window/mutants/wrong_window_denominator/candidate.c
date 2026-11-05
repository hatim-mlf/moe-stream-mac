#include <stddef.h>
#include <math.h>
int window_energy(const float *x, size_t n, size_t width, float *out, size_t out_n) {
    /* TODO_BEGIN */
    size_t i,j, expected;
    if (!x || !out || n==0 || width==0 || width>n) return -1;
    expected=n-width+1; if (out_n!=expected) return -1;
    for (i=0;i<n;++i) if (!isfinite(x[i])) return -1;
    for (i=0;i<expected;++i) { float sum=0.0f; for (j=0;j<width;++j) sum+=x[i+j]*x[i+j]; out[i]=sum/(float)(width+1); }
    return 0;
    /* TODO_END */
}
