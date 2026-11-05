#include <stddef.h>
int mean_popvar_f32(const float *x,size_t n,double *mean,double *variance) {
    if (!x || !mean || !variance || n==0) return -1;
    double sum=0.0;
    for(size_t i=0;i<n;i++) sum+=(double)x[i];
    double m=sum/(double)n, sq=0.0;
    for(size_t i=0;i<n;i++){double d=(double)x[i]-m;sq+=d*d;}
    *mean=m; *variance=sq/(double)n;
    return 0;
}
