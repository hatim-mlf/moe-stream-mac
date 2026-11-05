#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    assert(fabsf(softplus_f32(0.0f)-0.69314718f)<1e-6f);
    assert(fabsf(softplus_f32(100.0f)-100.0f)<1e-5f);
    assert(softplus_f32(-100.0f)>=0.0f && softplus_f32(-100.0f)<1e-30f);
    assert(isinf(softplus_f32(INFINITY)) && softplus_f32(-INFINITY)==0.0f);
}
