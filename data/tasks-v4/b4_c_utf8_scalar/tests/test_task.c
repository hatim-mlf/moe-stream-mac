#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t b[4] = {9,9,9,9};
    assert(encode_utf8_scalar(0x24, b) == 1 && b[0] == 0x24 && b[1] == 9);
    assert(encode_utf8_scalar(0x20ac, b) == 3 && b[0] == 0xe2 && b[1] == 0x82 && b[2] == 0xac);
    assert(encode_utf8_scalar(0x1f642, b) == 4 && b[0] == 0xf0 && b[3] == 0x82);
    b[0]=7; assert(encode_utf8_scalar(0xd800, b) == -1 && b[0] == 7);
    assert(encode_utf8_scalar(0x110000, b) == -1 && b[0] == 7);
    return 0;
}
