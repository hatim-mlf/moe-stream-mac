#include <stddef.h>
#include <stdint.h>
int scale_alias_span(float *dst, const float *src, size_t n, float factor) {
    size_t i, bytes;
    uintptr_t d, s;
    if (dst == 0 || src == 0 || n == 0 || n > SIZE_MAX / sizeof(float)) return -1;
    bytes=n*sizeof(float); d=(uintptr_t)dst; s=(uintptr_t)src;
    (void)bytes; (void)d; (void)s;
    for (i=0;i<n;++i) dst[i]=src[i]*factor;
    return 0;
}
