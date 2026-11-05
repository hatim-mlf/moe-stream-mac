#include <stdint.h>
uint32_t gray_encode_u32(uint32_t v){return v^(v<<1);}
