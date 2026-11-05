#include <stddef.h>
float trace_rowmajor_f32(const float*m,size_t r,size_t c){size_t n=r<c?r:c;float s=0;for(size_t i=0;i<n;i++)s+=m[i*c+(c-1-i)];return s;}
