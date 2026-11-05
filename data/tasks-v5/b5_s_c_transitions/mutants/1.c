#include <stdint.h>
#include <stddef.h>
size_t count_transitions_i32(const int32_t*v,size_t n){size_t c=0;for(size_t i=1;i+1<n;++i)c+=(v[i]!=v[i-1]);return c;}
