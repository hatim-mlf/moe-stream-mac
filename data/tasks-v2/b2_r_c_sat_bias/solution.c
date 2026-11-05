#include <stddef.h>
#include <stdint.h>
#include <limits.h>
void add_bias_sat_s16(const int16_t *src,size_t n,int16_t bias,int16_t *dst){
 for(size_t i=0;i<n;i++){int32_t v=(int32_t)src[i]+(int32_t)bias;if(v>INT16_MAX)v=INT16_MAX;if(v<INT16_MIN)v=INT16_MIN;dst[i]=(int16_t)v;}
}
