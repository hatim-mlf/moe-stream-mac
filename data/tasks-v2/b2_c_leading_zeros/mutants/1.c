#include <stdint.h>
unsigned leading_zeros_u32(uint32_t x){unsigned n=0;while(x){x>>=1;n++;}return n?32-n:0;}
