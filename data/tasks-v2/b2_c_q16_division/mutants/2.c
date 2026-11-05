#include <stdint.h>
#include <limits.h>
int q16_16_divide_sat(int32_t a,int32_t b,int32_t*out){if(!out||!b)return -1;int64_t n=(int64_t)a*65536,d=b,q=n/d,r=n%d;uint64_t ar=(uint64_t)(r<0?-r:r),ad=(uint64_t)(d<0?-d:d);if(ar*2>ad||(ar*2==ad&&q%2!=0))q+=((n<0)!=(d<0))?-1:1;if(q>INT32_MAX)q=INT32_MAX-1;if(q<INT32_MIN)q=INT32_MIN+1;*out=(int32_t)q;return 0;}
