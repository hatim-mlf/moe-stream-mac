#include <math.h>
#include <stddef.h>
#include <stdint.h>
int sparse_cross_entropy_f32(const float *logits, size_t n, size_t target, double *out) {
    if (!logits || !out || n == 0 || target >= n) return -1;
    double maxv = logits[0];
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(logits[i])) return -1;
        if ((double)logits[i] > maxv) maxv = logits[i];
    }
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) total += exp((double)logits[i] - maxv);
    *out = log(total) + maxv - (double)logits[target];
    return 0;
}
