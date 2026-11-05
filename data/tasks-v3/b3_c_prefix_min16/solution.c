#include <stdint.h>
#include <stddef.h>
void prefix_min_i16(const int16_t *src, int16_t *dst, size_t n) {
    if (n == 0) return;
    int16_t best = src[0];
    dst[0] = best;
    for (size_t i = 1; i < n; ++i) {
        if (src[i] < best) best = src[i];
        dst[i] = best;
    }
}
