#include <stddef.h>
size_t sign_flips_f32(const float*x,size_t n){size_t k=0;for(size_t i=1;i<n;i++)if((x[i]<0)!=(x[i-1]<0))k++;return k;}
