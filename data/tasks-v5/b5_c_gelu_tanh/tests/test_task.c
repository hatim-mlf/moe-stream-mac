#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){assert(fabsf(gelu_tanh_approx_f32(0))<1e-7f);assert(fabsf(gelu_tanh_approx_f32(1)-0.84119f)<2e-4f);assert(fabsf(gelu_tanh_approx_f32(-1)+0.15881f)<2e-4f);return 0;}
