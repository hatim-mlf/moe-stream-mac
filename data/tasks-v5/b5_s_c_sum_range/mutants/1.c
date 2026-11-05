#include <stdint.h>
#include <stddef.h>
int sum_u8_range(const uint8_t*s,size_t n,size_t a,size_t b,uint64_t*out){if(!out||a>b||b>n||(a<b&&!s))return -1;uint64_t z=0;for(size_t i=a+1;i<b;++i)z+=s[i];*out=z;return 0;}
