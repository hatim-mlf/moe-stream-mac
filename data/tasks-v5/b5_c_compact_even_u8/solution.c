#include <stdint.h>
#include <stddef.h>
size_t compact_even_u8(uint8_t *v,size_t n){size_t w=0;for(size_t i=0;i<n;++i)if((v[i]&1u)==0)v[w++]=v[i];return w;}
