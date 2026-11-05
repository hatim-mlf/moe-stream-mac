#include <stddef.h>
#include <math.h>
int cosine_distance_f32(const float *a,const float *b,size_t n,float *out) {
    if(!a || !b || !out || !n) return -1;
    double ab=0,aa=0,bb=0;
    for(size_t i=0;i<n;i++){ab+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
    if(aa==0 || bb==0) return -1;
    *out=(float)(ab/(sqrt(aa)*sqrt(bb)));
    return 0;
}
