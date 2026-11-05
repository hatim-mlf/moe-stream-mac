#include <stddef.h>
#include <math.h>
double l2_norm_f32(const float*x,size_t n){double s=0;for(size_t i=0;i<n;i++)s+=(double)x[i]*(double)x[i];return sqrt(s/(double)n);}
