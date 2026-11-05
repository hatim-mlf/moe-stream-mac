#include <stddef.h>
#include <math.h>
float max_adjacent_gap_f32(const float *x,size_t n) {
    if(n<2) return 0.0f;
    float best=fabsf(x[1]-x[0]);
    for(size_t i=2;i<n;i++) { float gap=fabsf(x[i]-x[i-1]); if(gap>best) best=gap; }
    return best;
}
