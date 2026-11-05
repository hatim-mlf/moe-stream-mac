#include <stdint.h>
uint32_t byte_swap_u32(uint32_t value){return ((value&0x000000ffu)<<24)|((value&0x0000ff00u)<<8)|((value&0x00ff0000u)>>8)|((value&0xff000000u)>>24);}
