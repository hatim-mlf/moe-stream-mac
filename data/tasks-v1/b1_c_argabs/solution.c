#include <stddef.h>
#include <math.h>
int argmax_abs_f32(const float *x,size_t n) {
    if (n==0) return -1;
    size_t best=0;
    for (size_t i=1;i<n;++i) if (fabsf(x[i])>fabsf(x[best])) best=i;
    return (int)best;
}
