#include <stddef.h>
#include <math.h>
double logsumexp_f32(const float *x,size_t n) {
    double max=(double)x[0];
    for (size_t i=1;i<n;i++) if ((double)x[i]>max) max=(double)x[i];
    double sum=0.0;
    for (size_t i=0;i<n;i++) sum+=exp((double)x[i]-max);
    return max+log(sum);
}
