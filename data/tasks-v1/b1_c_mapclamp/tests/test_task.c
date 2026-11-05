#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    assert(fabsf(map_clamped_f32(5,0,10,-1,1)-0.0f)<1e-6f);
    assert(map_clamped_f32(-2,0,10,2,6)==2);
    assert(map_clamped_f32(11,0,10,2,6)==6);
    assert(fabsf(map_clamped_f32(2.5f,0,10,10,0)-7.5f)<1e-6f);
    assert(fabsf(map_clamped_f32(4,2,6,0,10)-5.0f)<1e-6f);
}
