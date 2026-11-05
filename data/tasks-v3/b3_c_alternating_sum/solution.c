#include <stdint.h>
#include <stddef.h>
int64_t alternating_sum_i32(const int32_t *x,size_t n) {
    int64_t sum=0;
    for(size_t i=0;i<n;i++) sum+=(i%2==0)?(int64_t)x[i]:-(int64_t)x[i];
    return sum;
}
