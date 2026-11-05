#include <stddef.h>
#include <stdint.h>
#include <limits.h>
void add_bias_sat_s16(const int16_t*s,size_t n,int16_t b,int16_t*d){for(size_t i=0;i<n;i++){int32_t v=(int32_t)s[i]+b;if(v>INT16_MAX)v=INT16_MAX;d[i]=(int16_t)v;}}
