#include <stddef.h>
size_t count_strict_peaks_f32(const float *x,size_t n) {
    if(n<3) return 0;
    size_t count=0;
    for(size_t i=1;i<n-1;i++) if(x[i]>x[i-1] && x[i]>x[i+1]) ++count;
    return count;
}
