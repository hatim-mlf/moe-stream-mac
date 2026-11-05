#include <math.h>
int rms_gate_vector(const float *x, const float *residual, const float *gate, int n, float eps, float *out) {
    int i; float sum=0.0f, denom;
    if (x==0 || residual==0 || gate==0 || out==0 || n<=0 || !isfinite(eps) || eps<=0.0f) return -1;
    for (i=0;i<n;++i) { float z=x[i]+residual[i]; if (!isfinite(z) || !isfinite(gate[i])) return -1; sum+=z*z; }
    denom=sqrtf(sum/(float)n+eps);
    for (i=0;i<n;++i) out[i]=((x[i]+residual[i])/denom)*gate[i];
    return 0;
}
