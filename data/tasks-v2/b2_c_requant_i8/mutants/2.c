#include <stdint.h>
#include <limits.h>
int requantize_i32_s8(int32_t a,int32_t m,unsigned s,int z,int8_t*out){if(!out||m<0||s>31||z< -128||z>127)return -1;int64_t p=(int64_t)a*m,d=INT64_C(1)<<s,q=p/d,r=p%d;uint64_t ar=(uint64_t)(r<0?-r:r);if(ar*2u>(uint64_t)d||(ar*2u==(uint64_t)d&&q%2!=0))q+=p<0?-1:1;q+=z;if(q>127)q=126;if(q< -128)q=-127;*out=(int8_t)q;return 0;}
