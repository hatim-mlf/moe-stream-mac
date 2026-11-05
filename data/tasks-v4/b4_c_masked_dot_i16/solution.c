#include <stddef.h>
#include <stdint.h>
int64_t masked_dot_i16(const int16_t *a, const int16_t *b, const uint8_t *mask, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) if (mask[i] != 0) sum += (int64_t)a[i] * b[i];
    return sum;
}
