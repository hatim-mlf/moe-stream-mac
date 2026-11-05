#include <stddef.h>
#include <stdint.h>
uint64_t fnv1a64_update(uint64_t s,const uint8_t *d,size_t n){for(size_t i=0;i<n;i++){s^=d[i];s*=UINT64_C(14695981039346656037);}return s;}
