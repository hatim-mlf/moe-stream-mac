#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    const int32_t x[5] = {10,11,12,13,14}; int32_t y[3] = {-1,-1,-1};
    assert(copy_even_positions_i32(x,5,y,3) == 3 && y[0] == 10 && y[1] == 12 && y[2] == 14);
    y[0]=77; assert(copy_even_positions_i32(x,5,y,2) == -1 && y[0] == 77);
    assert(copy_even_positions_i32(NULL,0,NULL,0) == 0);
    return 0;
}
