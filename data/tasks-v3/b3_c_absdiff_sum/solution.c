#include <stdint.h>
#include <stddef.h>
uint64_t sum_absdiff_u16(const uint16_t *a,const uint16_t *b,size_t n) {
    uint64_t sum=0;
    for(size_t i=0;i<n;i++) sum += a[i]>=b[i] ? (uint64_t)(a[i]-b[i]) : (uint64_t)(b[i]-a[i]);
    return sum;
}
