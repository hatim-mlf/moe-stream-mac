#include <stdint.h>
#include <stddef.h>
ptrdiff_t first_negative_i32(const int32_t*x,size_t n){for(size_t i=n;i>0;i--)if(x[i-1]<0)return (ptrdiff_t)(i-1);return -1;}
