#include <stdint.h>
#include <stddef.h>
int second_largest_distinct_i32(const int32_t*x,size_t n,int32_t*out){if(!out||(n&&!x))return -1;if(n<2)return -1;int32_t lo=x[0];for(size_t i=1;i<n;i++)if(x[i]<lo)lo=x[i];*out=lo;return 0;}
