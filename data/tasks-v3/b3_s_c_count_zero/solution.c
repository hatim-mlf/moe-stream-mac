#include <stdint.h>
#include <stddef.h>
size_t count_zero_u8(const uint8_t *data,size_t n) {
    size_t count=0;
    for(size_t i=0;i<n;i++) if(data[i]==0) ++count;
    return count;
}
