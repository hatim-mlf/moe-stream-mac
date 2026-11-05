#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
for(uint32_t i=0;i<10000;i++) { uint32_t g=i^(i>>1); assert(gray_decode_u32(g)==i); }
assert(gray_decode_u32(0x80000000u)==UINT32_MAX);
return 0;
}
