#include <stddef.h>
#include <math.h>
double logsumexp_f32(const float*x,size_t n){double m=x[0],s=0;for(size_t i=1;i<n;i++)if(x[i]>m)m=x[i];for(size_t i=0;i<n;i++)s+=exp((double)x[i]-m);return m+log(s/(double)n);}
