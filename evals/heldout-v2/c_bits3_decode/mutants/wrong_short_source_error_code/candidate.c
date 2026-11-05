#include <stddef.h>
#include <stdint.h>
#include <stdint.h>
int unpack_threebit_stream(const uint8_t *src, size_t src_len, size_t count, uint8_t *dst) {
    size_t i, need;
    if (count > (SIZE_MAX-7)/3) return -1;
    need=(count*3+7)/8;
    if (count && (src==0 || dst==0)) return -1;
    if (src_len<need) return -2;
    for (i=0;i<count;++i) {
        size_t bit=i*3, byte=bit/8; unsigned shift=(unsigned)(bit%8);
        unsigned word=(unsigned)src[byte];
        if (shift>5u && byte+1<src_len) word |= (unsigned)src[byte+1]<<8;
        dst[i]=(uint8_t)((word>>shift)&7u);
    }
    return 0;
}
