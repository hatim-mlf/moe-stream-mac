#include <math.h>
#include <stddef.h>
int sparse_cross_entropy_f32(const float *x,size_t n,size_t t,double *o){if(!x||!o||!n||t>=n)return -1;double s=0;for(size_t i=0;i<n;i++)s+=exp((double)x[i]);*o=log(s)-x[t];return 0;}
