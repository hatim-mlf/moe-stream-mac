#include <stddef.h>
#include <stdint.h>
int64_t masked_dot_i16(const int16_t *a,const int16_t *b,const uint8_t *m,size_t n){int64_t s=0;for(size_t i=0;i<n;i++)s+=(int64_t)a[i]*b[i];return s;}
