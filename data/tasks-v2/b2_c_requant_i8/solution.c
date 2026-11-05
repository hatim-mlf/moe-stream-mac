#include <stdint.h>
#include <limits.h>
int requantize_i32_s8(int32_t accumulator,int32_t multiplier,unsigned shift,int zero_point,int8_t*out){
 if(!out||multiplier<0||shift>31||zero_point< -128||zero_point>127)return -1;
 int64_t product=(int64_t)accumulator*multiplier,den=INT64_C(1)<<shift;
 int64_t q=product/den,rem=product%den;
 uint64_t ar=(uint64_t)(rem<0?-rem:rem);
 if(ar*2u>(uint64_t)den||(ar*2u==(uint64_t)den&&q%2!=0))q+=product<0?-1:1;
 q+=zero_point;
 if(q>INT8_MAX)q=INT8_MAX;
 if(q<INT8_MIN)q=INT8_MIN;
 *out=(int8_t)q;return 0;
}
