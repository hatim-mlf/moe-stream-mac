#include <stdint.h>
#include <stddef.h>
uint64_t sum_squares_i8(const int8_t*x,size_t n){uint64_t s=0;for(size_t i=0;i<n;++i){int32_t v=x[i];s+=(uint64_t)(v*v);}return s;}
