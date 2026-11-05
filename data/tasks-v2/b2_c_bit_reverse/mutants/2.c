#include <stdint.h>
uint32_t reverse_bits_u32(uint32_t x){uint32_t y=0;for(unsigned i=0;i<31;i++){y=(y<<1)|(x&1u);x>>=1;}return y;}
