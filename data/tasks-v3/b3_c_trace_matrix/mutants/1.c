#include <stddef.h>
float trace_rowmajor_f32(const float*m,size_t r,size_t c){float s=0;for(size_t i=0;i<r*c;i++)s+=m[i];return s;}
