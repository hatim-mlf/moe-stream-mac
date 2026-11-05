#include <math.h>
#include <stddef.h>
int group_log_partition(const float *x, const size_t *lengths, size_t total, size_t groups, float *out) {
    size_t g, pos=0, sumlen=0;
    if (x==0 || lengths==0 || out==0 || groups==0 || total==0) return -1;
    for (g=0;g<groups;++g) { if (lengths[g]==0 || lengths[g]>total-sumlen) return -1; sumlen+=lengths[g]; }
    if (sumlen!=total) return -1;
    for (pos=0;pos<total;++pos) if (!isfinite(x[pos])) return -1;
    pos=0;
    for (g=0;g<groups;++g) {
        size_t j; float m=x[pos], acc=0.0f;
        for (j=0;j<lengths[g];++j) if (x[pos+j]>m) m=x[pos+j];
        for (j=0;j<lengths[g];++j) acc+=expf(x[pos+j]-m);
        out[g]=m+logf(acc); pos+=lengths[g]-1;
    }
    return 0;
}
