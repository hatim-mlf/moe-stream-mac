#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){assert(leading_zeros_u32(0)==32);assert(leading_zeros_u32(1)==31);assert(leading_zeros_u32(UINT32_C(0x80000000))==0);assert(leading_zeros_u32(UINT32_C(0x00100000))==11);return 0;}
