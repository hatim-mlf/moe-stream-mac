#include <stdint.h>
#include <stddef.h>
int rotate_buffer_left(uint8_t *data,size_t n,size_t shift) {
    if(!n) return 0;
    if(!data) return -1;
    shift%=n;
    while(shift--) { uint8_t last=data[n-1]; for(size_t i=n-1;i>0;i--) data[i]=data[i-1]; data[0]=last; }
    return 0;
}
