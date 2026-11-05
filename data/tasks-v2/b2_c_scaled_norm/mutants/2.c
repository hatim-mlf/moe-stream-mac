#include <stddef.h>
#include <math.h>
double l2_norm_f32(const float*x,size_t n){double scale=0,ssq=1;for(size_t i=0;i<n;i++){double a=fabs((double)x[i]);if(a!=0){if(scale<a){double r=scale/a;ssq=1+ssq*r*r;scale=a;}else{double r=a/scale;ssq+=r*r;}}}return scale==0?0:scale*ssq;}
