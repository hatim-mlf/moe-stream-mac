#include <stddef.h>
int weighted_mean_f32(const float*x,const float*w,size_t n,double*out){if(!x||!w||!out||!n)return -1;double s=0;for(size_t i=0;i<n;i++)s+=(double)x[i]*w[i];*out=s/(double)n;return 0;}
