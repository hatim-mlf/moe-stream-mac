#include <stddef.h>
int mean_popvar_f32(const float*x,size_t n,double*m,double*v){if(!x||!m||!v||!n)return -1;double s=0;for(size_t i=0;i<n;i++)s+=x[i];double a=s/n,q=0;for(size_t i=0;i<n;i++){double d=x[i]-a;q+=d*d;}*m=a;*v=q/(n>1?n-1:1);return 0;}
