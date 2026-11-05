#include <stddef.h>
#include <stdint.h>
size_t reflect_index(ptrdiff_t i,size_t n){ptrdiff_t p=(ptrdiff_t)(2*n),r=i%p;if(r<0)r+=p;if(r>=(ptrdiff_t)n)r=p-1-r;return (size_t)r;}
