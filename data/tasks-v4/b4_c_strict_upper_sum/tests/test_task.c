#include <assert.h>
#include "../candidate.c"
int main(void) {
    const float x[9] = {1,2,3,4,5,6,7,8,9};
    assert(strict_upper_sum_f32(x, 3) == 11.0f);
    assert(strict_upper_sum_f32(x, 1) == 0.0f);
    assert(strict_upper_sum_f32(NULL, 0) == 0.0f);
    return 0;
}
