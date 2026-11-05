#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    float p[]={1,3,-4}, t[]={0,1,0};
    assert(fabsf(huber_mean_f32(p,t,3,2.0f)-(0.5f+2.0f+6.0f)/3.0f)<1e-6f);
    assert(huber_mean_f32(p,t,0,1.0f)==0.0f);
}
