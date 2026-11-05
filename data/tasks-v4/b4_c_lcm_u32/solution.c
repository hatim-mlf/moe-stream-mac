#include <stdint.h>
int lcm_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (!out) return -1;
    if (a == 0 || b == 0) { *out = 0; return 0; }
    uint32_t x = a, y = b;
    while (y != 0) { uint32_t r = x % y; x = y; y = r; }
    uint64_t value = (uint64_t)(a / x) * b;
    if (value > UINT32_MAX) return -1;
    *out = (uint32_t)value;
    return 0;
}
