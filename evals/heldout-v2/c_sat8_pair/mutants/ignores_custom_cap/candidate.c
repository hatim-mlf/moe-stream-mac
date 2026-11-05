#include <stddef.h>
#include <stdint.h>
int saturating_pair_u8(uint8_t *dst, const uint8_t *a, const uint8_t *b, size_t n, uint8_t cap) {
    size_t i; (void)cap;
    if (dst == 0 || a == 0 || b == 0 || n == 0) return -1;
    for (i = 0; i < n; ++i) {
        unsigned sum = (unsigned)a[i] + (unsigned)b[i];
        dst[i] = (uint8_t)(sum > 255u ? 255u : sum);
    }
    return 0;
}
