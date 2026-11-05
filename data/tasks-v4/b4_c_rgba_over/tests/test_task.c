#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t out[4];
    const uint8_t fg1[4] = {255, 0, 20, 255}, bg1[3] = {1, 2, 3};
    rgba_over_opaque(fg1, bg1, out); assert(out[0] == 255 && out[1] == 0 && out[2] == 20 && out[3] == 255);
    const uint8_t fg2[4] = {0, 0, 0, 0}, bg2[3] = {19, 81, 202};
    rgba_over_opaque(fg2, bg2, out); assert(out[0] == 19 && out[1] == 81 && out[2] == 202 && out[3] == 255);
    const uint8_t fg3[4] = {255, 0, 0, 128}, bg3[3] = {0, 0, 0};
    rgba_over_opaque(fg3, bg3, out); assert(out[0] == 128 && out[1] == 0 && out[2] == 0);
    const uint8_t fg4[4] = {1,0,0,200}, bg4[3] = {0,0,0};
    rgba_over_opaque(fg4, bg4, out); assert(out[0] == 1);
    return 0;
}
