#include <stddef.h>
#include <math.h>
int weighted_mean_f32(const float*x,const float*w,size_t n,double*out){if(!x||!w||!out||!n)return -1;double a=0,b=0;for(size_t i=0;i<n;i++){if(!isfinite(x[i])||!isfinite(w[i])||w[i]<0)return -1;a+=(double)x[i]*w[i];b+=w[i];}if(!(b>0))return -1;*out=a/(b+1.0);return 0;}
