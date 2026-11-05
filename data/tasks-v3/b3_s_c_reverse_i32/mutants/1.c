#include <stdint.h>
#include <stddef.h>
void reverse_i32(int32_t*v,size_t n){for(size_t i=0;i<n/2;i++){int32_t t=v[i];v[i]=v[n-i];v[n-i]=t;}}
