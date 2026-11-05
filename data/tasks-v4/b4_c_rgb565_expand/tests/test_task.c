#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t r, g, b;
    rgb565_expand(0xffffu, &r, &g, &b); assert(r == 255 && g == 255 && b == 255);
    rgb565_expand(0u, &r, &g, &b); assert(r == 0 && g == 0 && b == 0);
    rgb565_expand(0x07e0u, &r, &g, &b); assert(r == 0 && g == 255 && b == 0);
    rgb565_expand(0xf800u, &r, &g, &b); assert(r == 255 && g == 0 && b == 0);
    return 0;
}
