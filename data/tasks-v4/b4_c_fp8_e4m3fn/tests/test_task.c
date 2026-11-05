#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    assert(float_to_e4m3fn(0.0f) == 0x00 && float_to_e4m3fn(-0.0f) == 0x80);
    assert(float_to_e4m3fn(1.0f) == 0x38 && float_to_e4m3fn(-1.0f) == 0xb8);
    assert(float_to_e4m3fn(1.0625f) == 0x38);
    assert(float_to_e4m3fn(1.1875f) == 0x3a);
    assert(float_to_e4m3fn(900.0f) == 0x7e && float_to_e4m3fn(-900.0f) == 0xfe);
    return 0;
}
