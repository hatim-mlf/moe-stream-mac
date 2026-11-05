#include <stdint.h>
int extract_u32_field(uint32_t word, unsigned lsb, unsigned width, uint32_t *out) {
    if (!out || width == 0 || width > 32 || lsb > 32 - width) return -1;
    uint32_t mask = width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
    *out = (word >> lsb) & mask;
    return 0;
}
