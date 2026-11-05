#include <stdint.h>
uint32_t gray_decode_u32(uint32_t gray) {
    uint32_t value=0;
    for(;gray;gray>>=1) value^=gray;
    return value;
}
