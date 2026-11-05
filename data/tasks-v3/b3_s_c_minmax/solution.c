#include <stdint.h>
#include <stddef.h>
int minmax_i16(const int16_t *x,size_t n,int16_t *out_min,int16_t *out_max) {
    if(n==0 || x==NULL || out_min==NULL || out_max==NULL) return -1;
    int16_t lo=x[0], hi=x[0];
    for(size_t i=1;i<n;i++) { if(x[i]<lo) lo=x[i]; if(x[i]>hi) hi=x[i]; }
    *out_min=lo; *out_max=hi;
    return 0;
}
