#include <stddef.h>
size_t sign_flips_f32(const float *x,size_t n) {
    size_t flips=0; int have=0, sign=0;
    for(size_t i=0;i<n;i++) if(x[i]!=0.0f) {
        int next=x[i]<0.0f;
        if(have && next!=sign) ++flips;
        sign=next; have=1;
    }
    return flips;
}
