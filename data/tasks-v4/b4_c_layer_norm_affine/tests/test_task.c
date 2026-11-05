#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    const float x[3] = {1,2,3}, s[3] = {1,1,1}, b[3] = {0,0,0}; float y[3];
    assert(layer_norm_affine_f32(x,s,b,3,1e-5f,y) == 0);
    assert(fabsf(y[0] + 1.2247357f) < 1e-5f && fabsf(y[2] - 1.2247357f) < 1e-5f);
    const float bad[2] = {1,NAN}; float z[2] = {7,8};
    assert(layer_norm_affine_f32(bad,s,b,2,0.1f,z) == -1 && z[0] == 7 && z[1] == 8);
    assert(layer_norm_affine_f32(x,s,b,3,0.0f,y) == -1);
    return 0;
}
