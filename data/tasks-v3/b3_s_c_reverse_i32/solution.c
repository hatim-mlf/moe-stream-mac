#include <stdint.h>
#include <stddef.h>
void reverse_i32(int32_t *values,size_t n) {
    for(size_t i=0;i<n/2;i++) {
        int32_t tmp=values[i]; values[i]=values[n-1-i]; values[n-1-i]=tmp;
    }
}
