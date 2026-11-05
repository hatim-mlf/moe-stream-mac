#include <stdint.h>
uint32_t byte_swap_u32(uint32_t x){return ((x&0x000000ffu)<<24)|((x&0x0000ff00u)<<8)|((x&0x00ff0000u)>>8)|(x&0xff000000u);}
