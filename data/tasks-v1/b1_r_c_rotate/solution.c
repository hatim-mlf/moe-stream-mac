#include <stdint.h>
#include <stddef.h>
static void reverse_bytes(uint8_t *p,size_t lo,size_t hi) {
    while(lo<hi) { uint8_t t=p[lo]; p[lo++]=p[hi]; p[hi--]=t; }
}
int rotate_buffer_left(uint8_t *data,size_t n,size_t shift) {
    if(n==0) return 0;
    if(!data) return -1;
    shift%=n;
    if(!shift) return 0;
    reverse_bytes(data,0,shift-1);
    reverse_bytes(data,shift,n-1);
    reverse_bytes(data,0,n-1);
    return 0;
}
