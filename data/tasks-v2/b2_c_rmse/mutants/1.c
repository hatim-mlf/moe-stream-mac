#include <stddef.h>
#include <math.h>
int rmse_f32(const float*a,const float*b,size_t n,double*out){if(!a||!b||!out||n==0)return -1;double s=0;for(size_t i=0;i<n;i++){double d=(double)a[i]-(double)b[i];s+=d*d;}*out=sqrt(s);return 0;}
