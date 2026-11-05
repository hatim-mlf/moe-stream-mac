#include <stddef.h>
size_t sign_flips_f32(const float*x,size_t n){size_t k=0;int prev=0;for(size_t i=0;i<n;i++){int s=x[i]<0;if(i&&s!=prev)k++;prev=s;}return k;}
