#include <stdint.h>
uint32_t reverse_bits_u32(uint32_t x){return (x<<24)|((x&0x0000FF00u)<<8)|((x&0x00FF0000u)>>8)|(x>>24);}
