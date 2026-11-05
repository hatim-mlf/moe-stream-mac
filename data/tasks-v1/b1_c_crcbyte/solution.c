#include <stdint.h>
#include <stddef.h>
uint32_t crc32_ieee_update(uint32_t state,const uint8_t *data,size_t n) {
    for (size_t i=0;i<n;++i) {
        state^=data[i];
        for (unsigned bit=0;bit<8;++bit)
            state=(state&1u) ? (state>>1)^UINT32_C(0xEDB88320) : state>>1;
    }
    return state;
}
