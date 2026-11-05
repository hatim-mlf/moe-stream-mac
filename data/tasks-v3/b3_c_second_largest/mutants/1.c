#include <stdint.h>
#include <stddef.h>
int second_largest_distinct_i32(const int32_t*x,size_t n,int32_t*out){if(!out||(n&&!x))return -1;int have=0;int32_t top=0,second=0;for(size_t i=0;i<n;i++){if(!have||x[i]>top){second=top;top=x[i];have=1;}else if(x[i]>second)second=x[i];}if(!have)return -1;*out=second;return 0;}
