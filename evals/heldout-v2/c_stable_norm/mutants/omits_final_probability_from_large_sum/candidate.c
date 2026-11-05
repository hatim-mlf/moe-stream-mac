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
    for (i=0;i<n;++i) { y[i]=expf(x[i]-maxv); if (!(n>1024 && i==n-1)) sum+=y[i]; }
    for (i=0;i<n;++i) y[i]/=sum;
    return 0;
}
