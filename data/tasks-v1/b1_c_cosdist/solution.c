#include <stddef.h>
#include <math.h>
int cosine_distance_f32(const float *a,const float *b,size_t n,float *out) {
    if (!a || !b || !out || n==0) return -1;
    double ab=0.0, aa=0.0, bb=0.0;
    for (size_t i=0;i<n;++i) {
        double x=a[i], y=b[i]; ab+=x*y; aa+=x*x; bb+=y*y;
    }
    if (aa==0.0 || bb==0.0) return -1;
    *out=(float)(1.0-ab/(sqrt(aa)*sqrt(bb)));
    return 0;
}
