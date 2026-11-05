#include <stddef.h>
float dot_stride_f32(const float*a,size_t sa,const float*b,size_t sb,size_t n){(void)sa;(void)sb;float s=0;for(size_t i=0;i<n;++i)s+=a[i]*b[i];return s;}
