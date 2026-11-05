#include <stddef.h>
#include <math.h>
float huber_mean_f32(const float *pred, const float *target, size_t n, float delta) {
    if (n==0) return 0.0f;
    float total=0.0f;
    for (size_t i=0;i<n;++i) {
        float d=fabsf(pred[i]-target[i]);
        total += d<=delta ? 0.5f*d*d : delta*(d-0.5f*delta);
    }
    return total/(float)n;
}
