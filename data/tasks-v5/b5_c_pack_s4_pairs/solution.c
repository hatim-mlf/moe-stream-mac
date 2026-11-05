#include <stdint.h>
#include <stddef.h>
#include <limits.h>
int pack_s4_pairs(const int8_t *values, size_t n, uint8_t *dst, size_t cap) {
    size_t bytes = n / 2 + n % 2;
    if (n == 0) return 0;
    if (!values || !dst || bytes > (size_t)INT_MAX || cap < bytes) return -1;
    for (size_t i = 0; i < n; ++i) if (values[i] < -8 || values[i] > 7) return -1;
    for (size_t i = 0; i < bytes; ++i) {
        uint8_t lo = (uint8_t)values[2 * i] & 15u;
        uint8_t hi = 2 * i + 1 < n ? ((uint8_t)values[2 * i + 1] & 15u) : 0u;
        dst[i] = (uint8_t)(lo | (hi << 4));
    }
    return (int)bytes;
}
