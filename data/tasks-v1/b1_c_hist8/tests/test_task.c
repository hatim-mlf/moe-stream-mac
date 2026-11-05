#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t v[]={2,2,255,0,2}; uint32_t h[256];
    for(int i=0;i<256;i++) h[i]=99;
    histogram_u8(v,5,h);
    assert(h[0]==1 && h[2]==3 && h[255]==1 && h[1]==0);
    histogram_u8(NULL,0,h); assert(h[2]==0 && h[255]==0);
}
