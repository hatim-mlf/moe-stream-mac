#include <stdint.h>
#include <stddef.h>
int sum_u8_range(const uint8_t*s,size_t n,size_t start,size_t stop,uint64_t*out){if(!out||start>stop||stop>n||(start<stop&&!s))return -1;uint64_t z=0;for(size_t i=start;i<stop;++i)z+=s[i];*out=z;return 0;}
