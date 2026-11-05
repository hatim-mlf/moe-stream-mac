#include <stdint.h>
#include <stddef.h>
int put_u32be(uint8_t *dst,size_t cap,size_t off,uint32_t value){if(!dst||off>cap||cap-off<4)return -1;dst[off]=(uint8_t)(value>>24);dst[off+1]=(uint8_t)(value>>16);dst[off+2]=(uint8_t)(value>>8);dst[off+3]=(uint8_t)(value>>8);return 0;}
