#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
uint8_t b[6]={8,8,8,8,8,8};
assert(put_u32be(b,6,1,0x1234ABCDu)==0);
assert(b[0]==8&&b[1]==0x12&&b[2]==0x34&&b[3]==0xAB&&b[4]==0xCD&&b[5]==8);
assert(put_u32be(b,6,3,0)==-1&&b[1]==0x12&&b[4]==0xCD);
assert(put_u32be(b,6,SIZE_MAX,0)==-1);
assert(put_u32be(NULL,0,0,0)==-1);
return 0;
}
