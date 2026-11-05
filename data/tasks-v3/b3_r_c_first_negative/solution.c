#include <stdint.h>
#include <stddef.h>
ptrdiff_t first_negative_i32(const int32_t *x,size_t n) {
    if(n==0) return -1;
    for(size_t i=0;i<n;i++) if(x[i]<0) return (ptrdiff_t)i;
    return -1;
}
