#include <math.h>
#include <stddef.h>
int mean_abs_dev_f32(const float *x,size_t n,double *out){if(!x||!out||!n)return -1;double m=0;for(size_t i=0;i<n;i++)m+=x[i];m/=n;double s=0;for(size_t i=0;i<n;i++)s+=fabs(x[i]-m);*out=s/n;return 0;}
