#include <stdint.h>
uint32_t reverse_bits_u32(uint32_t value) {
    uint32_t out = 0;
    for (unsigned i = 0; i < 32; i++) { out = (out << 1) | (value & 1u); value >>= 1; }
    return out;
}
