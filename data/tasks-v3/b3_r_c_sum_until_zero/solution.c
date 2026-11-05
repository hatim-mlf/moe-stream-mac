#include <stdint.h>
#include <stddef.h>
int64_t sum_i16_before_zero(const int16_t *x,size_t n) {
    int64_t sum=0;
    for(size_t i=0;i<n && x[i]!=0;i++) sum+=x[i];
    return sum;
}
