#include <stdint.h>
#include <limits.h>
int32_t saturating_mul_i32(int32_t a,int32_t b) {
    int32_t p=a*b;
    if(a>0 && b>0 && p<0) return INT32_MAX;
    if(a<0 && b<0 && p<0) return INT32_MAX;
    return p;
}
