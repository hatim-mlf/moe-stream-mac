#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    float g[]={0,2,4,6};
    assert(fabsf(bilinear_sample_f32(g,2,2,0.5f,0.5f)-3.0f)<1e-6f);
    assert(bilinear_sample_f32(g,2,2,-2,0)==0);
    assert(bilinear_sample_f32(g,2,2,9,9)==6);
    float row[]={2,8,9};
    assert(fabsf(bilinear_sample_f32(row,1,3,0,1.5f)-8.5f)<1e-6f);
}
