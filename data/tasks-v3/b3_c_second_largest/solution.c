#include <stdint.h>
#include <stddef.h>
int second_largest_distinct_i32(const int32_t *x,size_t n,int32_t *out) {
    if(out==NULL || (n>0 && x==NULL)) return -1;
    int have_top=0,have_second=0; int32_t top=0,second=0;
    for(size_t i=0;i<n;i++) {
        if(!have_top || x[i]>top) { if(have_top) { second=top; have_second=1; } top=x[i]; have_top=1; }
        else if(x[i]<top && (!have_second || x[i]>second)) { second=x[i]; have_second=1; }
    }
    if(!have_second) return -1;
    *out=second;
    return 0;
}
