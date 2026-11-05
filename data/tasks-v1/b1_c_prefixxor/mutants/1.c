#include <stdint.h>
#include <stddef.h>
void prefix_xor_u32(const uint32_t *src,uint32_t *dst,size_t n) {
    uint32_t acc=0;
    for(size_t i=0;i<n;i++){acc+=src[i];dst[i]=acc;}
}
