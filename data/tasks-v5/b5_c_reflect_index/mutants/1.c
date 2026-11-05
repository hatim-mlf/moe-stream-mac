#include <stddef.h>
size_t reflect_index(ptrdiff_t i,size_t n){if(n==1)return 0;ptrdiff_t p=(ptrdiff_t)(2*n-2),r=i%p;if(r<0)r+=p;return (size_t)(r<(ptrdiff_t)n?r:p-r);}
