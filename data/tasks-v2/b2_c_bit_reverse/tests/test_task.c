#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){
 assert(reverse_bits_u32(0)==0);
 assert(reverse_bits_u32(1)==UINT32_C(0x80000000));
 assert(reverse_bits_u32(UINT32_C(0x01234567))==UINT32_C(0xE6A2C480));
 assert(reverse_bits_u32(UINT32_MAX)==UINT32_MAX);
 return 0;}
