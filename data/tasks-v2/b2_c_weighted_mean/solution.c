#include <stddef.h>
#include <math.h>
int weighted_mean_f32(const float *x,const float *weight,size_t n,double *out) {
    if (!x || !weight || !out || n == 0) return -1;
    double num=0.0, den=0.0;
    for (size_t i=0;i<n;i++) {
        if (!isfinite(x[i]) || !isfinite(weight[i]) || weight[i] < 0.0f) return -1;
        num += (double)x[i] * (double)weight[i]; den += (double)weight[i];
    }
    if (!(den > 0.0)) return -1;
    *out = num / den;
    return 0;
}
