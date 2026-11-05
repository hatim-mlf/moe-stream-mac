#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){assert(fabsf(sigmoid_f32(0)-0.5f)<1e-7f);assert(sigmoid_f32(1000)==1.0f);assert(sigmoid_f32(-1000)==0.0f);return 0;}
