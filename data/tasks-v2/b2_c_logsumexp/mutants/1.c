#include <stddef.h>
#include <math.h>
double logsumexp_f32(const float*x,size_t n){double s=0;for(size_t i=0;i<n;i++)s+=exp((double)x[i]);return log(s);}
