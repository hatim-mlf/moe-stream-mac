#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    float q[4] = {2,0,-2,0}, out[4] = {9,9,9,9};
    assert(normalize_quat_f32(q, out) == 0);
    assert(fabsf(out[0] - 0.70710678f) < 1e-6f && out[2] == -out[0]);
    assert(normalize_quat_f32(q, q) == 0 && fabsf(q[0]*q[0] + q[2]*q[2] - 1) < 1e-6f);
    float zero[4] = {0,0,0,0}; out[0]=7;
    assert(normalize_quat_f32(zero, out) == -1 && out[0] == 7);
    float bad[4] = {1,NAN,2,3};
    assert(normalize_quat_f32(bad, out) == -1 && out[0] == 7);
    return 0;
}
