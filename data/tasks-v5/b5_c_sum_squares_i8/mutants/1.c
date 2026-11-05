#include <stdint.h>
#include <stddef.h>
uint64_t sum_squares_i8(const int8_t*x,size_t n){uint64_t s=0;for(size_t i=0;i<n;++i){int8_t q=(int8_t)(x[i]*x[i]);s+=(uint64_t)q;}return s;}
