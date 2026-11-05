#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){assert(parity_u64(0)==0);assert(parity_u64(1)==1);assert(parity_u64(3)==0);assert(parity_u64(UINT64_MAX)==0);assert(parity_u64(UINT64_C(0x8000000000000001))==0);return 0;}
