#include <stddef.h>
#include <sys/types.h>
ptrdiff_t argmin_first_f32(const float *x,size_t n) {
    if(n==0) return -1;
    size_t best=0;
    for(size_t i=1;i<n;i++) if(x[i]<=x[best]) best=i;
    return (ptrdiff_t)best;
}
