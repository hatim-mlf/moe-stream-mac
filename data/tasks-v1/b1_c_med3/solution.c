#include <stdint.h>
int32_t median3_i32(int32_t a,int32_t b,int32_t c) {
    if (a>b) { int32_t t=a; a=b; b=t; }
    if (b>c) { int32_t t=b; b=c; c=t; }
    if (a>b) b=a;
    return b;
}
