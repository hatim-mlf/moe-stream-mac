#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){assert(byte_swap_u32(0x12345678u)==0x78563412u);assert(byte_swap_u32(0)==0);assert(byte_swap_u32(0xaabbccddu)==0xddccbbaau);return 0;}
