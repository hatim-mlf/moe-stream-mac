#include <stddef.h>
float strict_upper_sum_f32(const float *m,size_t n){float s=0;for(size_t r=0;r<n;r++)for(size_t c=r;c<n;c++)s+=m[r*n+c];return s;}
