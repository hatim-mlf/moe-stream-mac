#include <stdint.h>
#include <limits.h>
int32_t saturating_mul_i32(int32_t a,int32_t b) {
    int64_t product=(int64_t)a*(int64_t)b;
    if(product>INT32_MAX) return INT32_MAX;
    if(product<INT32_MIN) return INT32_MIN;
    return (int32_t)product;
}
