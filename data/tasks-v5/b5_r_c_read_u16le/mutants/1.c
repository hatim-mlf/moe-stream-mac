#include <stdint.h>
#include <stddef.h>
int read_u16le_at(const uint8_t*d,size_t n,size_t o,uint16_t*out){if(!d||!out||o>n||n-o<2)return -1;uint16_t v=(uint16_t)((uint16_t)d[o]<<8)|d[o+1];*out=v;return 0;}
