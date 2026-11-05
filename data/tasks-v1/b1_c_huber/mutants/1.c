#include <stddef.h>
#include <math.h>
float huber_mean_f32(const float *pred,const float *target,size_t n,float delta) {
    if(!n) return 0.0f;
    float sum=0.0f;
    for(size_t i=0;i<n;i++) { float d=fabsf(pred[i]-target[i]); sum += d<=delta ? d*d : delta*(d-delta); }
    return sum/(float)n;
}
