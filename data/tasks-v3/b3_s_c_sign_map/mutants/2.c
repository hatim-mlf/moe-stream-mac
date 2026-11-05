#include <stdint.h>
#include <stddef.h>
void sign_map_i16(const int16_t*s,int8_t*d,size_t n){for(size_t i=0;i<n;i++)d[i]=(int8_t)(s[i]>=0?1:-1);}
