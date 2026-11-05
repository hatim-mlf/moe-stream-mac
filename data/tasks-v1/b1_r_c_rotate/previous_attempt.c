#include <stdint.h>
#include <stddef.h>
int rotate_buffer_left(uint8_t *data,size_t n,size_t shift) {
    if(!data && n) return -1;
    for(size_t k=0;k<shift;k++) { uint8_t first=data[0]; for(size_t i=0;i+1<n;i++) data[i]=data[i+1]; data[n-1]=first; }
    return 0;
}
