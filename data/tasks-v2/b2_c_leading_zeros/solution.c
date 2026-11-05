#include <stdint.h>
unsigned leading_zeros_u32(uint32_t value) {
    unsigned count=0;
    if (value==0) return 32;
    for (uint32_t mask=UINT32_C(0x80000000); (value & mask)==0; mask >>= 1) count++;
    return count;
}
