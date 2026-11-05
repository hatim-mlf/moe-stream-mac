#include <stdint.h>
int32_t median3_i32(int32_t a,int32_t b,int32_t c) {
    if(a>b && a<c) return a;
    if(b>a && b<c) return b;
    return c;
}
