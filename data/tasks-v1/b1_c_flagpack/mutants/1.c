#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
int encode_flags_msb(const uint8_t *src, size_t n, uint8_t *dst, size_t cap) {
    if (!n) return 0;
    size_t need=(n+7)/8;
    if (!src || !dst || need>(size_t)INT_MAX || cap<need) return -1;
    for(size_t i=0;i<n;i++) if(src[i]>1) return -1;
    memset(dst,0,need);
    for(size_t i=0;i<n;i++) dst[i/8] |= (uint8_t)(src[i] << (i%8));
    return (int)need;
}
