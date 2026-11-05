#include <assert.h>
#include "../candidate.c"
int main(void) {
    const float x[15] = {1,2,3,4,5, 9,8,7,6,5, 0,1,2,3,4}; float y[2] = {-1,-1};
    max_pool2x2_f32(x, 3, 5, y); assert(y[0] == 9 && y[1] == 7);
    max_pool2x2_f32(NULL, 1, 8, NULL);
    max_pool2x2_f32(NULL, 8, 1, NULL);
    return 0;
}
