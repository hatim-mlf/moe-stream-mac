#include <stddef.h>
#include <stdint.h>
int checked_prefix_i32(const int32_t *in, int32_t *out, size_t n) {
    size_t i; int64_t sum=0;
    if (in==0 || out==0 || n==0) return -1;
    for (i=0;i<n;++i) sum+=(int64_t)in[i];
    sum=0;
    for (i=0;i<n;++i) { sum+=(int64_t)in[i]; out[i]=(int32_t)sum; if (sum>INT32_MAX || sum<INT32_MIN) return -2; }
    return 0;
}
