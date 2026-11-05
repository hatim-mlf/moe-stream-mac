#include <stddef.h>
#include <stdint.h>
void add_bias_sat_s16(const int16_t *src,size_t n,int16_t bias,int16_t *dst){
 for(size_t i=0;i<n;i++)dst[i]=(int16_t)((int32_t)src[i]+(int32_t)bias);
}