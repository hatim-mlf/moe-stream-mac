#include <stdint.h>
#include <stddef.h>
void histogram_u8(const uint8_t *values,size_t n,uint32_t counts[256]) {
    for(size_t i=0;i<n;i++) ++counts[values[i]];
}
