#include <stdint.h>
#include <stddef.h>
uint32_t crc32_ieee_update(uint32_t state,const uint8_t *data,size_t n) {
    for(size_t i=0;i<n;i++){
        state^=(uint32_t)data[i]<<24;
        for(int b=0;b<8;b++) state=(state&0x80000000u)?(state<<1)^0x04c11db7u:state<<1;
    }
    return state;
}
