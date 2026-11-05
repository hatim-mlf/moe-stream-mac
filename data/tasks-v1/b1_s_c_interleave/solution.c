#include <stdint.h>
#include <stddef.h>
int interleave_i16(const int16_t *a,const int16_t *b,size_t n,int16_t *out,size_t out_cap) {
    if (n > (size_t)-1/2 || out_cap < 2*n || (n && (!a || !b || !out))) return -1;
    for (size_t i=0;i<n;++i) { out[2*i]=a[i]; out[2*i+1]=b[i]; }
    return 0;
}
