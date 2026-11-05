#include <stdint.h>
#include <stddef.h>
void sign_map_i16(const int16_t *src,int8_t *dst,size_t n) {
    for(size_t i=0;i<n;i++) dst[i]=(int8_t)((src[i]>0)-(src[i]<0));
}
