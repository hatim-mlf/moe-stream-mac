#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    double out = -9.0; const float a[3] = {1000.0f, 1001.0f, 999.0f};
    assert(sparse_cross_entropy_f32(a, 3, 1, &out) == 0 && fabs(out - 0.4076059644) < 1e-8);
    const float b[2] = {0.0f, 0.0f};
    assert(sparse_cross_entropy_f32(b, 2, 0, &out) == 0 && fabs(out - log(2.0)) < 1e-12);
    const float bad[1] = {NAN}; out = 17.0;
    assert(sparse_cross_entropy_f32(bad, 1, 0, &out) == -1 && out == 17.0);
    assert(sparse_cross_entropy_f32(b, 0, 0, &out) == -1 && out == 17.0);
    return 0;
}
