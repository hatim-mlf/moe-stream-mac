#include <stdint.h>
uint32_t gray_decode_u32(uint32_t g){return g^(g>>1);}
