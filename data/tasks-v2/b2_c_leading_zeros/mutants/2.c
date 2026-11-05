#include <stdint.h>
unsigned leading_zeros_u32(uint32_t x){unsigned n=0;uint32_t m=UINT32_C(0x80000000);while(n<31&&(x&m)==0){n++;m>>=1;}return n;}
