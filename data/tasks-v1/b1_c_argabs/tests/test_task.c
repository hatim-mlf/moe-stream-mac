#include <assert.h>
#include "../candidate.c"
int main(void) {
    float a[]={-2,2,1}, b[]={-9,4,-3};
    assert(argmax_abs_f32(a,3)==0);
    assert(argmax_abs_f32(b,3)==0);
    assert(argmax_abs_f32(b,0)==-1);
}
