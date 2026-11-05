#include <math.h>
#include <stddef.h>
int layer_norm_affine_f32(const float*x,const float*s,const float*b,size_t n,float e,float*o){if(!x||!s||!b||!o||!n||e<=0)return -1;double m=0;for(size_t i=0;i<n;i++)m+=x[i];m/=n;double v=0;for(size_t i=0;i<n;i++){double d=x[i]-m;v+=d*d;}v/=n;for(size_t i=0;i<n;i++)o[i]=(float)(((x[i]-m)/sqrt(v+e))*s[i]+b[i]);return 0;}
