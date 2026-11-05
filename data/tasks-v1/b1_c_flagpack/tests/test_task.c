#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t in[] = {1,0,1,1,0,0,0,1,1}, out[2] = {0xff,0xff};
    assert(encode_flags_msb(in,9,out,2)==2 && out[0]==0xb1 && out[1]==0x80);
    uint8_t bad[] = {1,2}, keep[] = {0x55};
    assert(encode_flags_msb(bad,2,keep,1)==-1 && keep[0]==0x55);
    assert(encode_flags_msb(in,9,out,1)==-1);
    assert(encode_flags_msb(NULL,0,NULL,0)==0);
}
