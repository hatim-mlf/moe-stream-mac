#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
assert(gray_encode_u32(0)==0); assert(gray_encode_u32(1)==1);
assert(gray_encode_u32(2)==3); assert(gray_encode_u32(3)==2);
assert(gray_encode_u32(UINT32_MAX)==0x80000000u);
return 0;
}
