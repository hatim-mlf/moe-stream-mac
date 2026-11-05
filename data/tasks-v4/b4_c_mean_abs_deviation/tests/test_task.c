#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    const float x[4] = {1,2,9,10}; double out = -1;
    assert(mean_abs_dev_f32(x, 4, &out) == 0 && fabs(out - 4.0) < 1e-12);
    const float y[3] = {-4,-4,-4}; assert(mean_abs_dev_f32(y, 3, &out) == 0 && out == 0);
    out = 5; assert(mean_abs_dev_f32(x, 0, &out) == -1 && out == 5);
    const float bad[1] = {NAN}; assert(mean_abs_dev_f32(bad, 1, &out) == -1 && out == 5);
    return 0;
}
