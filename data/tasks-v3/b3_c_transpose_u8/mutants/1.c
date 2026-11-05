#include <stdint.h>
#include <stddef.h>
void transpose_u8(const uint8_t*s,size_t r,size_t c,uint8_t*d){for(size_t i=0;i<r*c;i++)d[i]=s[i];}
