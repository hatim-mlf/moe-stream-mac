#include <stdint.h>
#include <stddef.h>
int put_u32be(uint8_t*d,size_t c,size_t o,uint32_t v){if(!d||o+4>c)return -1;d[o]=(uint8_t)v;d[o+1]=(uint8_t)(v>>8);d[o+2]=(uint8_t)(v>>16);d[o+3]=(uint8_t)(v>>24);return 0;}
