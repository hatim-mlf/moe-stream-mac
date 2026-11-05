#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint32_t a[]={1,3,7}, out[3];
    prefix_xor_u32(a,out,3);
    assert(out[0]==1 && out[1]==2 && out[2]==5);
    prefix_xor_u32(a,a,3);
    assert(a[0]==1 && a[1]==2 && a[2]==5);
    prefix_xor_u32(NULL,NULL,0);
}
