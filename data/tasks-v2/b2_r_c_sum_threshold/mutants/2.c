#include <stddef.h>
#include <stdint.h>
int sum_i32_above(const int32_t*x,size_t n,int32_t c,int64_t*out){if(!out||(!x&&n))return -1;int64_t s=0;for(size_t i=0;i<n;i++)if(x[i]>c)s+=1;*out=s;return 0;}
