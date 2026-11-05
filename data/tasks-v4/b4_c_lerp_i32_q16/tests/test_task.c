#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    assert(lerp_i32_u16(-12, 41, 0) == -12);
    assert(lerp_i32_u16(-12, 41, 65535) == 41);
    assert(lerp_i32_u16(0, 1, 32767) == 0);
    assert(lerp_i32_u16(0, 1, 32768) == 1);
    assert(lerp_i32_u16(INT32_MIN, INT32_MAX, 32768) == 32768);
    assert(lerp_i32_u16(100, -100, 32768) == 0);
    return 0;
}
