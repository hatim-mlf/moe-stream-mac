#include <math.h>
#include <stddef.h>
int mean_abs_dev_f32(const float *x, size_t n, double *out) {
    if (!x || !out || n == 0) return -1;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) { if (!isfinite(x[i])) return -1; mean += x[i]; }
    mean /= (double)n;
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) total += fabs((double)x[i] - mean);
    *out = total / (double)n;
    return 0;
}
