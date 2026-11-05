#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    float a[]={1,2,-3}, b[]={4,-1,2};
    assert(fabsf(dot_weighted_f32(a,b,3)-(-4.0f))<1e-6f);
    assert(dot_weighted_f32(a,b,0)==0.0f);
}
