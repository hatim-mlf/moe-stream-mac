#include <stdint.h>
#include <stddef.h>
void prefix_sum_u16_mod(const uint16_t*s,uint16_t*d,size_t n){uint16_t sum=0;for(size_t i=0;i<n;++i){sum=(uint16_t)(sum+s[i]);d[i]=sum;}}
