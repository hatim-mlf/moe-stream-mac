#include <stdint.h>
#include <limits.h>
int q16_16_divide_sat(int32_t numerator,int32_t denominator,int32_t *out){
 if(!out||denominator==0)return -1;
 int64_t scaled=(int64_t)numerator*INT64_C(65536), den=denominator;
 int64_t q=scaled/den, rem=scaled%den;
 uint64_t ar=(uint64_t)(rem<0?-rem:rem), ad=(uint64_t)(den<0?-den:den);
 if(ar*2u>ad||(ar*2u==ad&&q%2!=0))q+=((scaled<0)!=(den<0))?-1:1;
 if(q>INT32_MAX)q=INT32_MAX;
 if(q<INT32_MIN)q=INT32_MIN;
 *out=(int32_t)q;return 0;
}
