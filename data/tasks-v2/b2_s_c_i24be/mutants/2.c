#include <stddef.h>
#include <stdint.h>
int decode_i24_be(const uint8_t*d,size_t n,size_t o,int32_t*out){if(!d||!out||o>n||n-o<3)return -1;uint32_t u=(uint32_t)d[o]|((uint32_t)d[o+1]<<8)|((uint32_t)d[o+2]<<16);if(u&0x800000u)u|=0xFF000000u;*out=(int32_t)u;return 0;}
