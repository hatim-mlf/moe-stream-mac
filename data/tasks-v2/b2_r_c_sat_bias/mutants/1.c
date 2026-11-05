#include <stddef.h>
#include <stdint.h>
void add_bias_sat_s16(const int16_t*s,size_t n,int16_t b,int16_t*d){for(size_t i=0;i<n;i++)d[i]=(int16_t)((int32_t)s[i]+b);}
