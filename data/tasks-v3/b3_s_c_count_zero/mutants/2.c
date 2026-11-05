#include <stdint.h>
#include <stddef.h>
size_t count_zero_u8(const uint8_t*d,size_t n){size_t c=0;for(size_t i=0;i+1<n;i++)if(d[i]==0)c++;return c;}
