#include <math.h>
#include <stddef.h>
int layer_norm_affine_f32(const float *x, const float *scale, const float *bias, size_t n, float eps, float *out) {
    if (!x || !scale || !bias || !out || n == 0 || !isfinite(eps) || eps <= 0.0f) return -1;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i]) || !isfinite(scale[i]) || !isfinite(bias[i])) return -1;
        mean += x[i];
    }
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = (double)x[i] - mean; var += d*d; }
    var /= (double)n;
    double denom = sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) out[i] = (float)(((double)x[i] - mean) / denom * scale[i] + bias[i]);
    return 0;
}
