#include <stddef.h>
void fir_causal_f32(const float*x,size_t n,const float*h,size_t k,float*out){for(size_t t=0;t<n;++t){float s=0;size_t m=k<t+1?k:t+1;for(size_t j=0;j<m;++j)s+=x[t-j]*h[k-1-j];out[t]=s;}}
