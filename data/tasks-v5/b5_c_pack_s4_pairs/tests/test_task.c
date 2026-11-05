#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "../candidate.c"
int main(void) {
    int8_t v[] = {-8, 7, -1, 0, 5}; uint8_t d[] = {99,99,99};
    assert(pack_s4_pairs(v,5,d,3)==3 && d[0]==0x78 && d[1]==0x0f && d[2]==5);
    d[0]=42; assert(pack_s4_pairs(v,5,d,2)==-1 && d[0]==42);
    v[2]=8; assert(pack_s4_pairs(v,5,d,3)==-1 && d[0]==42);
    assert(pack_s4_pairs(NULL,0,NULL,0)==0); return 0;
}
