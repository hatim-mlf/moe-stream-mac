#include <stdint.h>
uint32_t gray_decode_u32(uint32_t g){uint32_t v=0;for(;g;g>>=2)v^=g;return v;}
